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

# import stable_pretraining as spt
# import stable_worldmodel as swm

import sys
@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
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
    # dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    for k, v in dataset[0].items():
        print(k, v.shape)


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
    # cfg["num_workers"] = 0
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    print("len(train_set):", len(train_set))

    print("len(train loader):", len(train))

    train_batch = next(iter(train))


    print("train batch type:", type(train_batch))


    for k, v in train_batch.items():
        if hasattr(v, "shape"):
            print("train", k, v.shape, v.dtype)
        else:
            print("train", k, type(v), v)


if __name__ == "__main__":
    run()


