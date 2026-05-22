from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
DECODER_DIR = ROOT / "decoder"
for path in (ROOT, DECODER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decoder_infer import load_visualization_decoder
from decoder_train import build_decoder_dataset
from plain_infer import build_model, load_checkpoint


LEWM_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\cache\你的run_id\lewm_best.ckpt"
DECODER_CKPT_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\decoder_ckpts\visualization_decoder_best.pt"
H5_DATA_PATH = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data"
SAVE_DIR = r"C:\Users\wangh\Desktop\world_model\lewd2\worldmodel_rollout_vis"

INFER_INDEX = 0
SAVE_GIF = True


def load_cfg_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "cfg" not in ckpt:
        raise KeyError("checkpoint does not contain cfg.")
    return OmegaConf.create(ckpt["cfg"])


def build_dataset(cfg):
    return build_decoder_dataset(cfg, H5_DATA_PATH)


def load_world_model(cfg, ckpt_path, device):
    world_model = build_model(cfg)
    return load_checkpoint(world_model, ckpt_path, device)


def load_decoder(ckpt_path, cfg, device):
    return load_visualization_decoder(
        ckpt_path=ckpt_path,
        device=device,
        cls_dim=cfg.wm.embed_dim,
        hidden_dim=256,
        image_size=cfg.img_size,
        patch_size=16,
        channels=3,
        depth=4,
        heads=8,
        dim_head=64,
    )


def denormalize_imagenet(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)

    shape = [1] * x.ndim
    shape[-3] = 3
    return (x * std.view(*shape) + mean.view(*shape)).clamp(0, 1)


def tensor_batch(sample, device):
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


@torch.no_grad()
def rollout_latent_with_dataset_actions(world_model, sample, cfg, device):
    batch = tensor_batch(sample, device)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    real_pixels = batch["pixels"]
    output = world_model.encode(batch)
    real_emb = output["emb"]
    actions = batch["action"]

    history_size = cfg.wm.history_size
    if real_emb.size(1) <= history_size:
        raise ValueError(
            f"T={real_emb.size(1)} <= history_size={history_size}; no rollout steps."
        )

    rollout_emb = real_emb[:, :history_size].clone()
    action_history = actions[:, :history_size].clone()

    for step in range(real_emb.size(1) - history_size):
        act_emb = world_model.action_encoder(action_history)
        pred_seq = world_model.predict(
            rollout_emb[:, -history_size:],
            act_emb[:, -history_size:],
        )

        rollout_emb = torch.cat([rollout_emb, pred_seq[:, -1:]], dim=1)

        next_action_idx = history_size + step
        next_action = actions[:, next_action_idx: next_action_idx + 1]
        action_history = torch.cat([action_history, next_action], dim=1)

    return {
        "real_pixels": real_pixels.detach(),
        "real_emb": real_emb.detach(),
        "rollout_emb": rollout_emb.detach(),
    }


@torch.no_grad()
def decode_latent_sequence(decoder, latent_seq):
    batch_size, num_steps, dim = latent_seq.shape
    decoded = decoder(latent_seq.reshape(batch_size * num_steps, dim))
    return decoded.reshape(batch_size, num_steps, *decoded.shape[1:])


def save_image_grids(real_pixels, generated_pixels, save_dir, prefix):
    real_vis = denormalize_imagenet(real_pixels.detach().cpu())
    gen_vis = denormalize_imagenet(generated_pixels.detach().cpu())

    outputs = {
        "real": real_vis,
        "generated_from_worldmodel": gen_vis,
        "comparison": torch.cat([real_vis, gen_vis], dim=0),
    }

    for name, tensor in outputs.items():
        path = save_dir / f"{prefix}_{name}.png"
        save_image(tensor, path, nrow=real_vis.size(0))
        print(f"Saved: {path}")


def save_comparison_gif(real_pixels, generated_pixels, save_path, fps=5):
    real_vis = denormalize_imagenet(real_pixels.detach().cpu())
    gen_vis = denormalize_imagenet(generated_pixels.detach().cpu())

    frames = []
    for real, generated in zip(real_vis, gen_vis):
        pair = torch.cat([real, generated], dim=-1)
        frame = pair.permute(1, 2, 0).numpy()
        frames.append((frame * 255).clip(0, 255).astype(np.uint8))

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved GIF: {save_path}")


def save_rollout_result(save_path, result):
    torch.save(
        {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in result.items()
        },
        save_path,
    )
    print(f"Saved tensor result: {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("========== World Model Rollout Visualization ==========")
    print(f"device            = {device}")
    print(f"LEWM_CKPT_PATH    = {LEWM_CKPT_PATH}")
    print(f"DECODER_CKPT_PATH = {DECODER_CKPT_PATH}")
    print(f"H5_DATA_PATH      = {H5_DATA_PATH}")
    print(f"INFER_INDEX       = {INFER_INDEX}")
    print(f"SAVE_DIR          = {SAVE_DIR}")

    cfg = load_cfg_from_ckpt(LEWM_CKPT_PATH)
    dataset = build_dataset(cfg)
    world_model = load_world_model(cfg, LEWM_CKPT_PATH, device)
    decoder = load_decoder(DECODER_CKPT_PATH, cfg, device)

    rollout = rollout_latent_with_dataset_actions(
        world_model=world_model,
        sample=dataset[INFER_INDEX],
        cfg=cfg,
        device=device,
    )

    generated_pixels = decode_latent_sequence(decoder, rollout["rollout_emb"])
    latent_mse = (rollout["rollout_emb"] - rollout["real_emb"]).pow(2).mean().item()

    print("\n========== Result ==========")
    print(f"real_pixels shape      = {tuple(rollout['real_pixels'].shape)}")
    print(f"real_emb shape         = {tuple(rollout['real_emb'].shape)}")
    print(f"rollout_emb shape      = {tuple(rollout['rollout_emb'].shape)}")
    print(f"generated_pixels shape = {tuple(generated_pixels.shape)}")
    print(f"latent_mse             = {latent_mse:.8f}")

    prefix = f"sample_{INFER_INDEX}"
    real_pixels_vis = rollout["real_pixels"][0]
    generated_pixels_vis = generated_pixels[0]

    save_image_grids(real_pixels_vis, generated_pixels_vis, save_dir, prefix)
    if SAVE_GIF:
        save_comparison_gif(
            real_pixels=real_pixels_vis,
            generated_pixels=generated_pixels_vis,
            save_path=save_dir / f"{prefix}_comparison.gif",
            fps=5,
        )

    save_rollout_result(
        save_dir / f"{prefix}_worldmodel_rollout_result.pt",
        {
            **rollout,
            "generated_pixels": generated_pixels,
            "latent_mse": latent_mse,
            "infer_index": INFER_INDEX,
            "lewm_ckpt": LEWM_CKPT_PATH,
            "decoder_ckpt": DECODER_CKPT_PATH,
        },
    )

    print("Visualization finished.")


if __name__ == "__main__":
    main()
