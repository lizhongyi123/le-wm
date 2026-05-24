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

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# def load_train_val(cfg, h5_path):
#
#     dataset = HDF5Dataset(**cfg.data.dataset, transform=None, h5_data_path=h5_path)
#     transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
#
#     with open_dict(cfg):
#         for col in cfg.data.dataset.keys_to_load:
#             if col.startswith("pixels"):
#                 continue
#
#             normalizer = get_column_normalizer(dataset, col, col)
#             transforms.append(normalizer)
#
#             setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
#
#     transform = Compose(*transforms)
#     dataset.transform = transform
#
#     rnd_gen = torch.Generator().manual_seed(cfg.seed)
#     train_set, val_set = random_split(
#         dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
#     )
#
#     train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
#     val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
#
#     return train, val
def load_train_val(
    cfg,
    h5_path,
    distributed=False,
    rank=0,
    world_size=1,
):
    """
    构建 train_loader 和 val_loader。

    单卡模式:
        train_loader: shuffle=True
        val_loader  : shuffle=False

    DDP 多卡模式:
        train_loader 使用 DistributedSampler
        val_loader   使用 DistributedSampler

    返回:
        train_loader
        val_loader
        train_sampler
        val_sampler
    """

    # ============================================================
    # 1. Dataset
    # ============================================================
    dataset = HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
        h5_data_path=h5_path,
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

    transform = Compose(*transforms)
    dataset.transform = transform

    # ============================================================
    # 2. Train / val split
    # ============================================================
    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    train_set, val_set = random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    # ============================================================
    # 3. DataLoader kwargs
    # ============================================================
    loader_kwargs = dict(cfg.loader)

    # 避免后面重复传参
    loader_kwargs.pop("shuffle", None)
    loader_kwargs.pop("drop_last", None)
    loader_kwargs.pop("generator", None)
    loader_kwargs.pop("sampler", None)

    # ============================================================
    # 4. Distributed mode
    # ============================================================
    if distributed:
        # --------------------------------------------------------
        # Train sampler / loader
        # --------------------------------------------------------
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            **loader_kwargs,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
        )

        # --------------------------------------------------------
        # Val sampler / loader
        # --------------------------------------------------------
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            **loader_kwargs,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
        )

    # ============================================================
    # 5. Single GPU / non-distributed mode
    # ============================================================
    else:
        train_sampler = None
        val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            **loader_kwargs,
            shuffle=True,
            drop_last=True,
            generator=rnd_gen,
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            **loader_kwargs,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader, train_sampler, val_sampler
