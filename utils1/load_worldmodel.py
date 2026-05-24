from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor

from my_swm import HDF5Dataset
from my_spt import Compose, vit_hf



def build_world_model(cfg):
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


def load_wm_checkpoint(model, ckpt_path, device):
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