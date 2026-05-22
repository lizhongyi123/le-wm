from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor

from my_swm import HDF5Dataset
from my_spt import Compose, vit_hf


# ============================================================
# 这里直接写死推理参数，不通过 Hydra 命令行新增参数
# ============================================================

CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\cache\你的run_id\lewm_best.ckpt"

H5_DATA_PATH = r"/train_data"

INFER_INDEX = 0

SAVE_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\infer_result.pt"


# ============================================================
# 工具函数
# ============================================================

def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_dataset(cfg):
    """
    和训练时保持一致的数据读取与 transform。
    """
    dataset = HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
        h5_data_path=H5_DATA_PATH,
    )

    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = Compose(*transforms)

    return dataset


def build_model(cfg):
    """
    和训练时完全一致地构建 JEPA world model。
    """
    encoder = vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        emb_dim=embed_dim,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    return world_model


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint 中没有 model_state_dict。当前 keys={list(ckpt.keys())}"
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"global_step: {ckpt.get('global_step', 'unknown')}")

    return model


# ============================================================
# 推理一：用真实 action 做一步 latent 预测
# ============================================================

@torch.no_grad()
def infer_one_sample(model, dataset, index, cfg, device):
    sample = dataset[index]

    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)  # 加 batch 维度
        else:
            batch[k] = v

    batch = move_to_device(batch, device)

    if "action" not in batch:
        raise KeyError("batch 中没有 action，无法做动作条件预测。")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = model.encode(batch)

    emb = output["emb"]          # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, A_emb)

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    if pred_emb.shape != tgt_emb.shape:
        raise RuntimeError(
            f"pred_emb.shape={pred_emb.shape}, tgt_emb.shape={tgt_emb.shape}"
        )

    mse = (pred_emb - tgt_emb).pow(2).mean()

    result = {
        "index": index,
        "mse": float(mse.item()),
        "emb_shape": tuple(emb.shape),
        "ctx_emb_shape": tuple(ctx_emb.shape),
        "pred_emb_shape": tuple(pred_emb.shape),
        "tgt_emb_shape": tuple(tgt_emb.shape),
        "emb": emb.detach().cpu(),
        "pred_emb": pred_emb.detach().cpu(),
        "tgt_emb": tgt_emb.detach().cpu(),
    }

    return result


# ============================================================
# 推理二：自回归 rollout
# ============================================================

@torch.no_grad()
def rollout_with_gt_actions(model, dataset, index, cfg, device, rollout_steps=None):
    """
    使用真实 action 做自回归 rollout。

    如果数据每条样本只有 4 帧，history_size=3，
    那么最多只能向后预测 1 步。

    如果想测更长 rollout，需要让 dataset.num_steps 更大。
    """
    sample = dataset[index]

    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    batch = move_to_device(batch, device)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = model.encode(batch)

    emb_gt = output["emb"]       # (B, T, D)
    action = batch["action"]     # (B, T, action_dim)

    B, T, D = emb_gt.shape
    ctx_len = cfg.wm.history_size

    if rollout_steps is None:
        rollout_steps = T - ctx_len

    rollout_steps = min(rollout_steps, T - ctx_len)

    emb_roll = emb_gt[:, :ctx_len].clone()
    act_roll = action[:, :ctx_len].clone()

    for i in range(rollout_steps):
        act_emb = model.action_encoder(act_roll)

        emb_trunc = emb_roll[:, -ctx_len:]
        act_trunc = act_emb[:, -ctx_len:]

        pred_seq = model.predict(emb_trunc, act_trunc)
        next_emb = pred_seq[:, -1:]

        emb_roll = torch.cat([emb_roll, next_emb], dim=1)

        next_action = action[:, ctx_len + i: ctx_len + i + 1]
        act_roll = torch.cat([act_roll, next_action], dim=1)

    target = emb_gt[:, : emb_roll.size(1)]
    rollout_mse = (emb_roll - target).pow(2).mean()

    result = {
        "index": index,
        "rollout_mse": float(rollout_mse.item()),
        "emb_roll_shape": tuple(emb_roll.shape),
        "emb_gt_shape": tuple(target.shape),
        "emb_roll": emb_roll.detach().cpu(),
        "emb_gt": target.detach().cpu(),
    }

    return result


# ============================================================
# Hydra 只读取原来的配置，不新增任何命令
# ============================================================

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Inference ==========")
    print(f"device      = {device}")
    print(f"CKPT_PATH   = {CKPT_PATH}")
    print(f"H5_DATA_PATH= {H5_DATA_PATH}")
    print(f"INFER_INDEX = {INFER_INDEX}")
    print(f"SAVE_PATH   = {SAVE_PATH}")

    dataset = build_dataset(cfg)

    model = build_model(cfg)
    model = load_checkpoint(model, CKPT_PATH, device)

    one_step_result = infer_one_sample(
        model=model,
        dataset=dataset,
        index=INFER_INDEX,
        cfg=cfg,
        device=device,
    )

    rollout_result = rollout_with_gt_actions(
        model=model,
        dataset=dataset,
        index=INFER_INDEX,
        cfg=cfg,
        device=device,
    )

    print("\n========== One-step Prediction ==========")
    print(f"sample index   = {one_step_result['index']}")
    print(f"latent mse     = {one_step_result['mse']:.8f}")
    print(f"emb shape      = {one_step_result['emb_shape']}")
    print(f"ctx_emb shape  = {one_step_result['ctx_emb_shape']}")
    print(f"pred_emb shape = {one_step_result['pred_emb_shape']}")
    print(f"tgt_emb shape  = {one_step_result['tgt_emb_shape']}")

    print("\n========== Rollout ==========")
    print(f"rollout mse    = {rollout_result['rollout_mse']:.8f}")
    print(f"emb_roll shape = {rollout_result['emb_roll_shape']}")
    print(f"emb_gt shape   = {rollout_result['emb_gt_shape']}")

    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "one_step": one_step_result,
            "rollout": rollout_result,
            "ckpt_path": str(CKPT_PATH),
            "infer_index": INFER_INDEX,
        },
        save_path,
    )

    print(f"\nSaved inference result to: {save_path}")


if __name__ == "__main__":
    run()