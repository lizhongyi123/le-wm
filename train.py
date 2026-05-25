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
import torch.distributed as dist
from utils1.load_worldmodel import build_world_model, load_wm_checkpoint
from utils1.load_dataset import load_train_val

def setup_distributed():
    """
    单卡 python 运行:
        distributed=False

    多卡 torchrun 运行:
        torchrun 会自动设置:
            RANK
            WORLD_SIZE
            LOCAL_RANK
    """
    if "RANK" not in os.environ:
        return False, 0, 1, 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    distributed, rank, world_size, local_rank = setup_distributed()
    distributed = None
    script_directory = os.path.dirname(os.path.abspath(__file__))

    base_run_dir = os.path.join(script_directory, "cache")
    resume_path =os.path.join(script_directory, "cache", 20260522_180443, "lewm_best.ckpt")

    try:
        if distributed:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h5_path = r"C:\Users\wangh\Desktop\world_model\le-wm\train_data"

        if is_main_process():
            print("========== Train LeWM ==========")
            print(f"distributed = {distributed}")
            print(f"rank        = {rank}")
            print(f"world_size  = {world_size}")
            print(f"local_rank  = {local_rank}")
            print(f"device      = {device}")
            print(f"h5_path     = {h5_path}")

        # --------------------------------------------------------
        # 1. Dataset
        # --------------------------------------------------------
        train_loader, val_loader, train_sampler, val_sampler = load_train_val(
            cfg=cfg,
            h5_path=h5_path,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        if is_main_process():
            print(f"train_loader batches = {len(train_loader)}")
            print(f"val_loader batches   = {len(val_loader)}")

        # --------------------------------------------------------
        # 2. Build model
        # --------------------------------------------------------
        world_model = build_world_model(cfg).to(device)

        sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

        # --------------------------------------------------------
        # 3. DDP wrap
        # --------------------------------------------------------
        if distributed:
            world_model = DDP(
                world_model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )

        # --------------------------------------------------------
        # 4. Run directory
        # --------------------------------------------------------


        if distributed:
            if rank == 0:
                run_id_obj = [cfg.get("subdir") or datetime.now().strftime("%Y%m%d_%H%M%S")]
            else:
                run_id_obj = [None]

            dist.broadcast_object_list(run_id_obj, src=0)
            run_id = run_id_obj[0]
        else:
            run_id = cfg.get("subdir") or datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = base_run_dir / run_id

        if is_main_process():
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
                OmegaConf.save(cfg, f)

            print(f"run_dir = {run_dir}")

        if distributed:
            dist.barrier()

        # --------------------------------------------------------
        # 5. Trainer
        # --------------------------------------------------------
        plain_trainer = PlainLeWMTrainer(
            model=world_model,
            sigreg=sigreg,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            cfg=cfg,
            run_dir=run_dir,
            output_model_name=cfg.output_model_name,
            device=device,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        # 是否重新训练
      # resume_path = r"C:\Users\wangh\Desktop\world_model\le-wm\cache\xxxx\lewm_last.ckpt"

        if resume_path is not None:
            plain_trainer.load_checkpoint(resume_path)

        plain_trainer.fit()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    run()