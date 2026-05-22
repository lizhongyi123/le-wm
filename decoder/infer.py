# infer.py
import torch
from torchvision.utils import save_image
from omegaconf import OmegaConf

from decoder_infer import load_visualization_decoder, reconstruct_with_decoder
from decoder_train import (
    build_decoder_dataset,
    build_encoder_from_cfg,
    load_encoder_from_lewm_checkpoint,
)


# ============================================================
# 需要你修改的路径
# ============================================================

LEWM_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\cache\你的run_id\lewm_best.ckpt"

DECODER_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\decoder_ckpts\visualization_decoder_best.pt"

H5_DATA_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data"

INFER_INDEX = 0

SAVE_PT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\decoder_infer_result.pt"

SAVE_IMAGE_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\decoder_reconstruction.png"


# ============================================================
# 反归一化 ImageNet 图像，便于保存查看
# ============================================================

def denormalize_imagenet(x):
    """
    x: (B, 3, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)
    return x


def load_cfg_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "cfg" not in ckpt:
        raise KeyError("checkpoint 中没有 cfg。")

    cfg = OmegaConf.create(ckpt["cfg"])
    return cfg


def build_dataset(cfg):
    return build_decoder_dataset(cfg, H5_DATA_PATH)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========== Decoder Inference ==========")
    print(f"device            = {device}")
    print(f"LEWM_CKPT_PATH    = {LEWM_CKPT_PATH}")
    print(f"DECODER_CKPT_PATH = {DECODER_CKPT_PATH}")
    print(f"H5_DATA_PATH      = {H5_DATA_PATH}")
    print(f"INFER_INDEX       = {INFER_INDEX}")

    cfg = load_cfg_from_ckpt(LEWM_CKPT_PATH)

    dataset = build_dataset(cfg)

    encoder = build_encoder_from_cfg(cfg)
    encoder = load_encoder_from_lewm_checkpoint(
        encoder=encoder,
        ckpt_path=LEWM_CKPT_PATH,
        device=device,
    )

    decoder = load_visualization_decoder(
        ckpt_path=DECODER_CKPT_PATH,
        device=device,
        cls_dim=192,
        hidden_dim=256,
        image_size=cfg.img_size,
        patch_size=16,
        channels=3,
        depth=4,
        heads=8,
        dim_head=64,
    )

    sample = dataset[INFER_INDEX]

    pixels = sample["pixels"]

    # pixels: (T, C, H, W)，这里加 batch 维
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(0)  # (1, T, C, H, W)

    pixels = pixels.to(device).float()

    recon = reconstruct_with_decoder(
        encoder=encoder,
        decoder=decoder,
        pixels=pixels,
        device=device,
    )

    torch.save(
        {
            "pixels": pixels.detach().cpu(),
            "recon": recon.detach().cpu(),
            "infer_index": INFER_INDEX,
            "lewm_ckpt": LEWM_CKPT_PATH,
            "decoder_ckpt": DECODER_CKPT_PATH,
        },
        SAVE_PT_PATH,
    )

    # 保存一张对比图
    # 如果是 (B,T,C,H,W)，取第一条序列的所有帧
    if pixels.ndim == 5:
        raw = pixels[0]
        rec = recon[0]
    else:
        raw = pixels
        rec = recon

    raw_vis = denormalize_imagenet(raw)
    rec_vis = denormalize_imagenet(rec)

    # 拼成：上面原图，下面重建图
    compare = torch.cat([raw_vis, rec_vis], dim=0)

    save_image(
        compare,
        SAVE_IMAGE_PATH,
        nrow=raw_vis.size(0),
    )

    print(f"Saved tensor result to: {SAVE_PT_PATH}")
    print(f"Saved image result to: {SAVE_IMAGE_PATH}")
    print(f"raw shape   = {tuple(pixels.shape)}")
    print(f"recon shape = {tuple(recon.shape)}")


if __name__ == "__main__":
    main()
