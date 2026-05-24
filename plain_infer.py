from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor

from my_swm import HDF5Dataset
from my_spt import Compose, vit_hf
from utils1.load_dataset import load_train_val
from utils1.load_worldmodel import load_wm_checkpoint, build_world_model
import copy
# ============================================================
# 这里直接写死推理参数，不通过 Hydra 命令行新增参数
# ============================================================

CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\cache\20260522_180443\lewm_best.ckpt"


INFER_INDEX = 0

# SAVE_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\infer_result.pt"


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



# ============================================================
# 推理二：自回归 rollout
# ============================================================

@torch.no_grad()
def rollout_with_gt_actions(model, loader, index, cfg, device, rollout_steps=None):
    """
    从 DataLoader 里取一个 batch，
    选其中第 index 个样本，
    使用真实 action block 做自回归 latent rollout。
    """

    model.eval()

    # 1. 取一个 batch
    batch = next(iter(loader))
    batch = move_to_device(batch, device)

    if "action" not in batch:
        raise KeyError("batch 中没有 action。")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

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

    # 5. 自回归预测
    for i in range(rollout_steps):
        # 当前使用最近 ctx_len 个 latent
        ctx_emb = emb_roll[:, -ctx_len:]     # [1, ctx_len, D]

        # 对应的真实 action block
        ctx_action = action[:, i:i + ctx_len]  # [1, ctx_len, action_dim]

        # action 编码
        ctx_act_emb = model.action_encoder(ctx_action)

        # 预测序列
        pred_seq = model.predict(ctx_emb, ctx_act_emb)

        # 取最后一个，作为下一跳 latent
        next_emb = pred_seq[:, -1:]          # [1, 1, D]

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

    # 6. 计算整体 rollout mse
    target = emb_gt[:, :emb_roll.size(1)]

    # 包含初始 context 的整体误差
    rollout_mse_all = (emb_roll - target).pow(2).mean()

    # 只计算预测部分误差
    rollout_mse_pred = (
        emb_roll[:, ctx_len:] - target[:, ctx_len:]
    ).pow(2).mean()

    result = {
        "index": index,
        "rollout_steps": rollout_steps,
        "rollout_mse_all": float(rollout_mse_all.item()),
        "rollout_mse_pred": float(rollout_mse_pred.item()),
        "step_mse": step_mse_list,
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

    print(f"INFER_INDEX = {INFER_INDEX}")
    # print(f"SAVE_PATH   = {SAVE_PATH}")

    h5_path = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data"

    cfg.data.dataset.num_steps = 20
    cfg.data.dataset.frameskip = 5  #跳过多少步
    train, val = load_train_val(cfg, h5_path)

    model = build_world_model(cfg)
    model = load_wm_checkpoint(model, CKPT_PATH, device)

    rollout_result = rollout_with_gt_actions(
        model=model,
        loader=val,
        index=0,
        cfg=cfg,
        device=device,
    )

    #
    # save_path = Path(SAVE_PATH)
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # torch.save(
    #     {
    #         "one_step": one_step_result,
    #         "rollout": rollout_result,
    #         "ckpt_path": str(CKPT_PATH),
    #         "infer_index": INFER_INDEX,
    #     },
    #     save_path,
    # )

    # print(f"\nSaved inference result to: {save_path}")


if __name__ == "__main__":
    run()