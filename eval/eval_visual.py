from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from utils1.load_dataset import load_train_val
from utils1.load_worldmodel import load_wm_checkpoint, build_world_model

from decoder.decoder_model import VisualizationDecoder
from utils1.unnormalize_imagenet import unnormalize_imagenet
from utils1.net_work_setting import  decoder_cfg
# ============================================================
# 这里直接写死推理参数，不通过 Hydra 命令行新增参数
# ============================================================

CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\cache\20260522_180443\lewm_best.ckpt"

DECODER_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\decoder\cache\20260523_234916\visualization_decoder_best.pt"

H5_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\train_data"

SAVE_DIR = r"C:\Users\wangh\Desktop\world_model\le-wm\eval\compare"

INFER_INDEX = 0
MAX_SAVE_IMAGES = 20


# ============================================================
# 工具函数
# ============================================================

def move_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)

            if k == "pixels":
                v = v.float()

            if k == "action":
                v = torch.nan_to_num(v.float(), 0.0)

            out[k] = v
        else:
            out[k] = v

    return out


def load_decoder_checkpoint(decoder, ckpt_path, device):
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "decoder_state_dict" not in ckpt:
        raise KeyError(
            f"Decoder checkpoint 中没有 decoder_state_dict，当前 keys={list(ckpt.keys())}"
        )

    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)

    print(f"Loaded decoder checkpoint: {ckpt_path}")

    if "epoch" in ckpt:
        print(f"decoder epoch: {ckpt['epoch']}")

    if "best_loss" in ckpt:
        print(f"decoder best_loss: {ckpt['best_loss']}")

    return decoder


# ============================================================
# 自回归 latent rollout + decoder 图像对比
# ============================================================

@torch.no_grad()
def rollout_with_gt_actions(
    model,
    decoder,
    loader,
    index,
    cfg,
    device,
    rollout_steps=None,
    save_dir=None,
    max_save_images=20,
):
    """
    从 DataLoader 里取一个 batch，
    选其中第 index 个样本，
    使用真实 action block 做自回归 latent rollout。

    然后使用 decoder 将：
        1. 真实 latent
        2. 自回归预测 latent

    解码为图像，并与真实图像进行比较。
    """

    model.eval()
    decoder.eval()

    # 1. 取一个 batch
    batch = next(iter(loader))
    batch = move_to_device(batch, device)

    if "action" not in batch:
        raise KeyError("batch 中没有 action。")

    if "pixels" not in batch:
        raise KeyError("batch 中没有 pixels。")

    # 2. 从 batch 中取第 index 个样本，保留 batch 维度
    sample = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            sample[k] = v[index:index + 1]
        else:
            sample[k] = v

    # sample["pixels"]: [1, T, C, H, W]
    # sample["action"]: [1, T, action_dim]

    # 3. 编码真实整段序列，得到 ground-truth latent
    output = model.encode(sample)

    if "emb" not in output:
        raise KeyError(f"model.encode(sample) 没有返回 emb，当前 keys={list(output.keys())}")

    emb_gt = output["emb"]          # [1, T, D]
    action = sample["action"]       # [1, T, action_dim]

    B, T, D = emb_gt.shape
    ctx_len = cfg.wm.history_size

    if T <= ctx_len:
        raise ValueError(
            f"T={T} 太短，history_size={ctx_len}，无法 rollout。"
        )

    if rollout_steps is None:
        rollout_steps = T - ctx_len

    rollout_steps = min(rollout_steps, T - ctx_len)

    # 4. 初始上下文使用真实 latent
    emb_roll = emb_gt[:, :ctx_len].clone()   # [1, ctx_len, D]

    step_mse_list = []

    # 5. 自回归预测 latent
    for i in range(rollout_steps):
        # 当前使用最近 ctx_len 个 latent
        ctx_emb = emb_roll[:, -ctx_len:]       # [1, ctx_len, D]

        # 对应的真实 action block
        ctx_action = action[:, i:i + ctx_len]  # [1, ctx_len, action_dim]

        # action 编码
        ctx_act_emb = model.action_encoder(ctx_action)

        # 预测序列
        pred_seq = model.predict(ctx_emb, ctx_act_emb)

        # 取最后一个，作为下一跳 latent
        next_emb = pred_seq[:, -1:]            # [1, 1, D]

        # 拼接到 rollout 序列后面
        emb_roll = torch.cat([emb_roll, next_emb], dim=1)

        # 当前步对应的真实目标
        gt_next = emb_gt[:, ctx_len + i:ctx_len + i + 1]

        step_mse = (next_emb - gt_next).pow(2).mean()
        step_mse_list.append(float(step_mse.item()))

        print(
            f"step {i:03d}: "
            f"predict latent index {ctx_len + i}, "
            f"mse = {step_mse.item():.8f}"
        )

    # 6. 计算 latent 误差
    target_emb = emb_gt[:, :emb_roll.size(1)]  # [1, T_roll, D]  #真实数据

    rollout_mse_all = (emb_roll - target_emb).pow(2).mean()   #推理数据

    rollout_mse_pred = (
        emb_roll[:, ctx_len:] - target_emb[:, ctx_len:]
    ).pow(2).mean()

    print("\n========== Latent Rollout Result ==========")
    print(f"rollout_mse_all  = {rollout_mse_all.item():.8f}")
    print(f"rollout_mse_pred = {rollout_mse_pred.item():.8f}")

    # ========================================================
    # 7. 使用 decoder 将自回归 latent 和真实 latent 解码为图像
    # ========================================================

    # emb_roll   : [1, T_roll, D]
    # target_emb : [1, T_roll, D]
    recon_roll = decoder(emb_roll)      # [1, T_roll, C, H, W]  #推理的
    recon_gt = decoder(target_emb)      # [1, T_roll, C, H, W]  #真实的

    # 真实图像，只取和 emb_roll 等长的部分
    pixels_gt = sample["pixels"][:, :emb_roll.size(1)]  # [1, T_roll, C, H, W]

    if recon_roll.shape != pixels_gt.shape:
        raise ValueError(
            f"recon_roll shape {tuple(recon_roll.shape)} != "
            f"pixels_gt shape {tuple(pixels_gt.shape)}"
        )

    if recon_gt.shape != pixels_gt.shape:
        raise ValueError(
            f"recon_gt shape {tuple(recon_gt.shape)} != "
            f"pixels_gt shape {tuple(pixels_gt.shape)}"
        )

    # 图像误差
    img_l1_roll_vs_real = F.l1_loss(recon_roll, pixels_gt)
    img_mse_roll_vs_real = F.mse_loss(recon_roll, pixels_gt)

    img_l1_gtlatent_vs_real = F.l1_loss(recon_gt, pixels_gt)
    img_mse_gtlatent_vs_real = F.mse_loss(recon_gt, pixels_gt)

    print("\n========== Decoder Image Comparison ==========")
    print("rollout latent -> decoder image vs real image:")
    print(f"  L1  = {img_l1_roll_vs_real.item():.8f}")
    print(f"  MSE = {img_mse_roll_vs_real.item():.8f}")

    print("gt latent -> decoder image vs real image:")
    print(f"  L1  = {img_l1_gtlatent_vs_real.item():.8f}")
    print(f"  MSE = {img_mse_gtlatent_vs_real.item():.8f}")

    # ========================================================
    # 8. 保存图像对比
    # ========================================================

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # [1, T, C, H, W] -> [T, C, H, W]
        real_img = pixels_gt[0].detach().cpu()
        gt_dec_img = recon_gt[0].detach().cpu()
        roll_dec_img = recon_roll[0].detach().cpu()

        real_img = unnormalize_imagenet(real_img)
        gt_dec_img = unnormalize_imagenet(gt_dec_img)
        roll_dec_img = unnormalize_imagenet(roll_dec_img)

        diff_img = (roll_dec_img - real_img).abs()

        n = min(max_save_images, real_img.size(0))

        real_save = real_img[:n]
        gt_dec_save = gt_dec_img[:n]
        roll_dec_save = roll_dec_img[:n]
        diff_save = diff_img[:n]

        save_image(
            real_save,
            save_dir / "real_image.png",
            nrow=n,
            normalize=True,
        )

        save_image(
            gt_dec_save,
            save_dir / "decoder_from_gt_latent.png",
            nrow=n,
            normalize=True,
        )

        save_image(
            roll_dec_save,
            save_dir / "decoder_from_rollout_latent.png",
            nrow=n,
            normalize=True,
        )

        save_image(
            diff_save,
            save_dir / "diff_rollout_vs_real.png",
            nrow=n,
            normalize=True,
        )

        # 第一行：真实图像
        # 第二行：真实 latent 经 decoder 重建图像
        # 第三行：自回归 latent 经 decoder 重建图像
        # 第四行：自回归重建图和真实图的差异
        compare = torch.cat(
            [real_save, gt_dec_save, roll_dec_save, diff_save],
            dim=0,
        )

        save_image(
            compare,
            save_dir / "compare_real_gtlatent_rollout_diff.png",
            nrow=n,
            normalize=True,
        )

        print(f"\nSaved decoder comparison images to: {save_dir}")
        print("real_image.png: 真实图像")
        print("decoder_from_gt_latent.png: 真实 latent 经 decoder 重建")
        print("decoder_from_rollout_latent.png: 自回归 latent 经 decoder 重建")
        print("diff_rollout_vs_real.png: 自回归重建图与真实图差异")
        print("compare_real_gtlatent_rollout_diff.png: 四行对比图")

    result = {
        "index": index,
        "rollout_steps": rollout_steps,

        "rollout_mse_all": float(rollout_mse_all.item()),
        "rollout_mse_pred": float(rollout_mse_pred.item()),
        "step_mse": step_mse_list,

        "emb_roll_shape": tuple(emb_roll.shape),
        "emb_gt_shape": tuple(target_emb.shape),

        "img_l1_roll_vs_real": float(img_l1_roll_vs_real.item()),
        "img_mse_roll_vs_real": float(img_mse_roll_vs_real.item()),
        "img_l1_gtlatent_vs_real": float(img_l1_gtlatent_vs_real.item()),
        "img_mse_gtlatent_vs_real": float(img_mse_gtlatent_vs_real.item()),

        "emb_roll": emb_roll.detach().cpu(),
        "emb_gt": target_emb.detach().cpu(),
    }

    return result


# ============================================================
# Hydra 只读取原来的配置，不新增任何命令
# ============================================================

@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Autoregressive Inference + Decoder ==========")
    print(f"device            = {device}")
    print(f"CKPT_PATH         = {CKPT_PATH}")
    print(f"DECODER_CKPT_PATH = {DECODER_CKPT_PATH}")
    print(f"H5_PATH           = {H5_PATH}")
    print(f"SAVE_DIR          = {SAVE_DIR}")
    print(f"INFER_INDEX       = {INFER_INDEX}")

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    cfg.data.dataset.num_steps = 20
    cfg.data.dataset.frameskip = 5

    train, val = load_train_val(cfg, H5_PATH)

    print(f"train batches = {len(train)}")
    print(f"val batches   = {len(val)}")

    # --------------------------------------------------------
    # 2. Build and load LeWM world model
    # --------------------------------------------------------
    model = build_world_model(cfg)
    model = load_wm_checkpoint(model, CKPT_PATH, device)

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # 3. Build and load decoder
    # --------------------------------------------------------
    latent_dim = cfg.wm.get("embed_dim", 192)

    model_cfg = dict(decoder_cfg["model"])
    model_cfg["latent_dim"] = cfg.wm.get("embed_dim", model_cfg.get("latent_dim", 192))
    model_cfg["image_size"] = cfg.img_size

    decoder = VisualizationDecoder(**model_cfg).to(device)


    decoder = load_decoder_checkpoint(
        decoder=decoder,
        ckpt_path=DECODER_CKPT_PATH,
        device=device,
    )

    decoder.eval()

    for p in decoder.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # 4. Rollout + decoder comparison
    # --------------------------------------------------------
    rollout_result = rollout_with_gt_actions(
        model=model,
        decoder=decoder,
        loader=val,
        index=INFER_INDEX,
        cfg=cfg,
        device=device,
        save_dir=SAVE_DIR,
        max_save_images=MAX_SAVE_IMAGES,
    )

    print("\n========== Summary ==========")
    for k, v in rollout_result.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: tensor shape {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    run()