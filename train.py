import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl

import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from plain_trainer import PlainLeWMTrainer
from my_swm import HDF5Dataset
from my_spt import Compose, ToImage, Resize, random_split, vit_hf
import my_spt as spt
import my_swm as swm

from datetime import datetime
from pathlib import Path



@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################
    # cfg.data.dataset.num_steps = 4
    # cfg.data.dataset.frameskip = 5  #跳过多少步
    # cfg.data.dataset.name = "pusht_expert_train"
    # cfg.data.dataset.keys_to_load = ["pixels", "action", "proprio", "state"]
    # cfg.data.dataset.keys_to_cache = ["action", "proprio", "state"]


    h5_path = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data"
    dataset = HDF5Dataset(**cfg.data.dataset, transform=None, h5_data_path=h5_path)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

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

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
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


    base_run_dir = Path("C:/Users/wangh/Desktop/world_model/le-wm/cache")

    run_id = cfg.get("subdir") or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_run_dir / run_id


    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    plain_trainer = PlainLeWMTrainer(
        model=world_model,
        sigreg=sigreg,
        train_loader=train,
        val_loader=val,
        cfg=cfg,
        run_dir=run_dir,
        output_model_name=cfg.output_model_name,
    )
    #是否重新训练
    resume_path = None

    if resume_path is not None:
        plain_trainer.load_checkpoint(resume_path)

    plain_trainer.fit()

    return


if __name__ == "__main__":
    run()
