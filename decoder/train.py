# train_decoder_from_lewm.py
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from decoder_train import (
    build_decoder_dataset,
    build_encoder_from_cfg,
    load_encoder_from_lewm_checkpoint,
    train_visualization_decoder,
)


# ============================================================
# 需要你修改的路径
# ============================================================

LEWM_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\cache\你的run_id\lewm_best.ckpt"

H5_DATA_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data"

DECODER_SAVE_DIR = r"C:\Users\wangh\Desktop\world_model\lewd2\decoder_ckpts"

DECODER_EPOCHS = 20
DECODER_LR = 1e-4


# ============================================================
# 从 LeWM checkpoint 读取 cfg
# ============================================================

def load_cfg_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cfg" not in ckpt:
        raise KeyError(
            "checkpoint 中没有 cfg。请确认 PlainLeWMTrainer.save_checkpoint 保存了 cfg。"
        )

    cfg = OmegaConf.create(ckpt["cfg"])
    return cfg


# ============================================================
# 构建数据集和 dataloader
# ============================================================

def build_dataset_and_loader(cfg):
    dataset = build_decoder_dataset(cfg, H5_DATA_PATH)

    loader_cfg = dict(cfg.loader)

    # Windows 下建议保守一点
    batch_size = loader_cfg.get("batch_size", 32)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, dataloader


# ============================================================
# 只加载 LeWM encoder，decoder 入口不构建 world model
# ============================================================

# ============================================================
# 主流程
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========== Train Visualization Decoder ==========")
    print(f"device           = {device}")
    print(f"LEWM_CKPT_PATH   = {LEWM_CKPT_PATH}")
    print(f"H5_DATA_PATH     = {H5_DATA_PATH}")
    print(f"DECODER_SAVE_DIR = {DECODER_SAVE_DIR}")

    cfg = load_cfg_from_ckpt(LEWM_CKPT_PATH)

    dataset, dataloader = build_dataset_and_loader(cfg)

    encoder = build_encoder_from_cfg(cfg)
    encoder = load_encoder_from_lewm_checkpoint(
        encoder=encoder,
        ckpt_path=LEWM_CKPT_PATH,
        device=device,
    )

    # 关键：decoder 的 encoder 来自训练好的 LeWM
    decoder = train_visualization_decoder(
        encoder=encoder,
        dataloader=dataloader,
        device=device,
        epochs=DECODER_EPOCHS,
        lr=DECODER_LR,
        save_dir=DECODER_SAVE_DIR,
        output_name="visualization_decoder",
        resume_from=None,
        cls_dim=192,
        hidden_dim=256,
        image_size=cfg.img_size,
        patch_size=16,
        channels=3,
        depth=4,
        heads=8,
        dim_head=64,
    )

    print("Decoder training finished.")


if __name__ == "__main__":
    main()
