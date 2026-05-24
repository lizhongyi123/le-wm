from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import save_image

from utils1.load_dataset import load_train_val
from utils1.load_worldmodel import build_world_model, load_wm_checkpoint

from decoder_model import VisualizationDecoder
from utils1.unnormalize_imagenet import unnormalize_imagenet
from utils1.net_work_setting import decoder_cfg
# ============================================================
# Paths
# ============================================================

LEWM_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\cache\20260522_180443\lewm_best.ckpt"

DECODER_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\decoder\cache\20260523_234916\visualization_decoder_best.pt"

H5_PATH = r"C:\Users\wangh\Desktop\world_model\le-wm\train_data"

SAVE_DIR = r"C:\Users\wangh\Desktop\world_model\le-wm\decoder\decoder_infer_result"

INFER_INDEX = 0
MAX_SAVE_IMAGES = 8


# ============================================================
# Helper
# ============================================================

def move_batch_to_device(batch, device):
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

    if "pixels" not in out:
        raise KeyError(f"batch 中没有 pixels，当前 keys={list(out.keys())}")

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


@torch.no_grad()
def infer_decoder_once(
    world_model,
    decoder,
    loader,
    index,
    device,
    save_dir,
    max_save_images=8,
):
    world_model.eval()
    decoder.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 取一个 batch
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    # 2. 取其中一个样本，保留 batch 维度
    sample = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            sample[k] = v[index:index + 1]
        else:
            sample[k] = v

    # sample["pixels"]: [1, T, C, H, W]
    real_pixel = sample["pixels"]

    # 3. 用 frozen LeWM encoder 得到 latent
    info = world_model.encode(sample)

    if "emb" not in info:
        raise KeyError(f"world_model.encode(sample) 没有返回 emb，当前 keys={list(info.keys())}")

    latent = info["emb"]

    print(f"target shape = {tuple(real_pixel.shape)}")
    print(f"latent shape = {tuple(latent.shape)}")


    latent_flat = latent   #潜变量


    # 5. decoder 重建图像
    recon_flat = decoder(latent_flat)

    print(f"recon_flat shape = {tuple(recon_flat.shape)}")


    # 6. 计算误差
    loss_l1 = F.l1_loss(recon_flat, real_pixel)
    loss_mse = F.mse_loss(recon_flat, real_pixel)

    print("\n========== Decoder Inference Result ==========")
    print(f"L1  loss = {loss_l1.item():.8f}")
    print(f"MSE loss = {loss_mse.item():.8f}")

    # 7. 保存图像
    real_pixel = rearrange(real_pixel, "b t c h w -> (b t) c h w")
    recon_img = rearrange(recon_flat, "b t c h w -> (b t) c h w")
    diff_img = (recon_img - real_pixel).abs()

    n = min(max_save_images, real_pixel.size(0))

    target_save = real_pixel[:n].detach().cpu()
    recon_save = recon_img[:n].detach().cpu()
    diff_save = diff_img[:n].detach().cpu()

    target_save = unnormalize_imagenet(target_save)
    recon_save = unnormalize_imagenet(recon_save)
    diff_save = unnormalize_imagenet(diff_save)

    save_image(
        target_save,
        save_dir / "real_pixel.png",
        nrow=n,
        normalize=True,
    )

    save_image(
        recon_save,
        save_dir / "recon.png",
        nrow=n,
        normalize=True,
    )

    save_image(
        diff_save,
        save_dir / "diff.png",
        nrow=n,
        normalize=True,
    )

    compare = torch.cat([target_save, recon_save, diff_save], dim=0)

    save_image(
        compare,
        save_dir / "compare_target_recon_diff.png",
        nrow=n,
        normalize=True,
    )

    print(f"\nSaved images to: {save_dir}")
    print("target.png: 真实图像")
    print("recon.png : decoder 重建图像")
    print("diff.png  : 绝对误差图")
    print("compare_target_recon_diff.png: 第一行真实图，第二行重建图，第三行误差图")

    result = {
        "l1_loss": float(loss_l1.item()),
        "mse_loss": float(loss_mse.item()),
        "target_shape": tuple(real_pixel.shape),
        "recon_shape": tuple(recon_flat.shape),
        "latent_shape": tuple(latent_flat.shape),
    }

    return result


# ============================================================
# Main
# ============================================================

@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Decoder Simple Inference ==========")
    print(f"device            = {device}")
    print(f"LEWM_CKPT_PATH    = {LEWM_CKPT_PATH}")
    print(f"DECODER_CKPT_PATH = {DECODER_CKPT_PATH}")
    print(f"INFER_INDEX       = {INFER_INDEX}")

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------
    cfg.data.dataset.num_steps = 20
    cfg.data.dataset.frameskip = 5

    train_loader, val_loader = load_train_val(cfg, H5_PATH)

    print(f"train_loader batches = {len(train_loader)}")
    print(f"val_loader batches   = {len(val_loader)}")

    # --------------------------------------------------------
    # 2. Build and load LeWM
    # --------------------------------------------------------
    world_model = build_world_model(cfg)

    world_model = load_wm_checkpoint(
        model=world_model,
        ckpt_path=LEWM_CKPT_PATH,
        device=device,
    )

    world_model.to(device)
    world_model.eval()

    for p in world_model.parameters():
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

    # --------------------------------------------------------
    # 4. Inference
    # --------------------------------------------------------
    result = infer_decoder_once(
        world_model=world_model,
        decoder=decoder,
        loader=val_loader,
        index=INFER_INDEX,
        device=device,
        save_dir=SAVE_DIR,
        max_save_images=MAX_SAVE_IMAGES,
    )

    print("\n========== Summary ==========")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run()