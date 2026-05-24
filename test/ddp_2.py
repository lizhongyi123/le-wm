# decoder_train_ddp.py
from pathlib import Path
from datetime import datetime
import os

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from omegaconf import OmegaConf

from utils1.load_dataset import load_train_val
from utils1.load_worldmodel import build_world_model, load_wm_checkpoint

from decoder_model import (
    VisualizationDecoder,
    LeWMEncodeWrapper,
    DecoderTrainingModel,
)

from utils1.net_work_setting import decoder_cfg


# ============================================================
# DDP helper
# ============================================================

def setup_distributed():
    """
    单卡 python 运行时:
        distributed=False

    torchrun 多卡运行时:
        torchrun 会自动设置:
            RANK
            WORLD_SIZE
            LOCAL_RANK
    """
    if "RANK" not in os.environ:
        return False, 0, 1, 0

    dist.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_ddp(model):
    """
    DDP 包装后，真实模型在 model.module 里。
    单卡时没有 module。
    """
    return model.module if hasattr(model, "module") else model


def reduce_mean_tensor(x, device):
    """
    用于多卡统计平均 loss。
    单卡时直接返回。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    x = x.detach().clone().to(device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


# ============================================================
# Batch helper
# ============================================================

def move_batch_to_device(batch, device):
    """
    把整个 batch 移动到 device。

    保留:
        batch["pixels"]
        batch["action"]
        以及其他可能的字段
    """
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch to be dict, got {type(batch)}")

    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)

            if k == "pixels":
                out[k] = out[k].float()

            if k == "action":
                out[k] = torch.nan_to_num(out[k].float(), 0.0)
        else:
            out[k] = v

    if "pixels" not in out:
        raise KeyError(f"batch 中没有 'pixels'。当前 keys={list(out.keys())}")

    return out


# ============================================================
# Decoder-only checkpoint
# ============================================================

def save_decoder_checkpoint(
    save_path,
    decoder,
    optimizer,
    epoch,
    global_step,
    best_loss,
    avg_loss,
    cfg=None,
    decoder_cfg_dict=None,
):
    """
    只保存 decoder 和 decoder optimizer。

    不保存 LeWM encoder / world_model。
    world_model 由 lewm_ckpt_path 单独管理。

    注意:
    DDP 下 decoder 是 DDP 包装对象，保存时需要 unwrap。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    decoder_to_save = unwrap_ddp(decoder)

    ckpt = {
        "decoder_state_dict": decoder_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "avg_loss": avg_loss,
    }

    if cfg is not None:
        ckpt["wm_cfg"] = OmegaConf.to_container(cfg, resolve=True)

    if decoder_cfg_dict is not None:
        ckpt["decoder_cfg"] = decoder_cfg_dict

    torch.save(ckpt, save_path)


def load_decoder_checkpoint(
    ckpt_path,
    decoder,
    optimizer=None,
    device="cuda",
):
    """
    只恢复 decoder 和 decoder optimizer。

    不恢复 LeWM encoder / world_model。

    DDP 下 decoder 可能是 DDP 包装对象，因此加载时也要 unwrap。
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    required_keys = [
        "decoder_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "best_loss",
    ]

    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        raise KeyError(
            f"Decoder checkpoint 缺少必要字段: {missing_keys}. "
            f"当前 keys={list(ckpt.keys())}"
        )

    decoder_to_load = unwrap_ddp(decoder)
    decoder_to_load.load_state_dict(ckpt["decoder_state_dict"], strict=True)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        for state in optimizer.state.values():
            for _, v in state.items():
                if torch.is_tensor(v):
                    v.data = v.data.to(device)

    start_epoch = int(ckpt["epoch"])
    global_step = int(ckpt["global_step"])
    best_loss = float(ckpt["best_loss"])

    if is_main_process():
        print(f"Loaded decoder checkpoint: {ckpt_path}")
        print(f"Resume from epoch: {start_epoch}")
        print(f"Global step: {global_step}")
        print(f"Best loss: {best_loss}")

    return start_epoch, global_step, best_loss


# ============================================================
# Train function
# ============================================================

def train_visualization_decoder(
    model,
    decoder,
    optimizer,
    dataloader,
    cfg,
    decoder_cfg_dict,
    device="cuda",
    epochs=20,
    save_dir="decoder_ckpts",
    output_name="visualization_decoder",
    resume_from=None,
    save_every_steps=None,
    distributed=False,
):
    """
    训练 visualization decoder。

    注意:
    - model 是 DecoderTrainingModel
    - LeWM encoder 已经在 main() 中冻结
    - decoder 是唯一训练对象
    - DDP 只包 decoder
    - checkpoint 只由 rank=0 保存
    """

    save_dir = Path(save_dir)

    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    last_path = save_dir / f"{output_name}_last.pt"
    best_path = save_dir / f"{output_name}_best.pt"

    model.to(device)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    # --------------------------------------------------------
    # Resume decoder only
    # --------------------------------------------------------
    if resume_from is not None and str(resume_from).lower() not in ["", "none", "null"]:
        start_epoch, global_step, best_loss = load_decoder_checkpoint(
            ckpt_path=resume_from,
            decoder=decoder,
            optimizer=optimizer,
            device=device,
        )

    if distributed:
        # 确保所有 rank 都完成 resume 后再开始训练
        dist.barrier()

    if start_epoch >= epochs:
        if is_main_process():
            print(f"start_epoch={start_epoch} >= epochs={epochs}, no training needed.")
        return decoder

    # --------------------------------------------------------
    # Train loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        if distributed and hasattr(dataloader, "sampler"):
            dataloader.sampler.set_epoch(epoch)

        model.decoder.train()
        model.lewm_encoder.eval()

        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0

        if is_main_process():
            pbar = tqdm(
                dataloader,
                desc=f"Decoder Train Epoch {epoch + 1}/{epochs}",
                dynamic_ncols=True,
            )
        else:
            pbar = dataloader

        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            output = model.compute_loss(batch)
            loss = output["loss"]

            loss.backward()
            optimizer.step()

            global_step += 1

            loss_value = float(output["loss"].item())
            l1_value = float(output["loss_l1"].item())
            l2_value = float(output["loss_l2"].item())

            total_loss += loss_value
            total_l1 += l1_value
            total_l2 += l2_value

            if is_main_process():
                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.6f}",
                        "l1": f"{l1_value:.6f}",
                        "l2": f"{l2_value:.6f}",
                    }
                )

            if save_every_steps is not None and save_every_steps > 0:
                if global_step % save_every_steps == 0:
                    if is_main_process():
                        save_decoder_checkpoint(
                            save_path=last_path,
                            decoder=decoder,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            best_loss=best_loss,
                            avg_loss=loss_value,
                            cfg=cfg,
                            decoder_cfg_dict=decoder_cfg_dict,
                        )

        n = max(len(dataloader), 1)

        avg_loss_local = torch.tensor(total_loss / n, device=device)
        avg_l1_local = torch.tensor(total_l1 / n, device=device)
        avg_l2_local = torch.tensor(total_l2 / n, device=device)

        avg_loss = reduce_mean_tensor(avg_loss_local, device).item()
        avg_l1 = reduce_mean_tensor(avg_l1_local, device).item()
        avg_l2 = reduce_mean_tensor(avg_l2_local, device).item()

        if is_main_process():
            print(
                f"\nEpoch [{epoch + 1}/{epochs}]\n"
                f"  loss = {avg_loss:.6f}\n"
                f"  l1   = {avg_l1:.6f}\n"
                f"  l2   = {avg_l2:.6f}\n"
            )

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if is_main_process():
            save_decoder_checkpoint(
                save_path=last_path,
                decoder=decoder,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_loss=best_loss,
                avg_loss=avg_loss,
                cfg=cfg,
                decoder_cfg_dict=decoder_cfg_dict,
            )

            if is_best:
                save_decoder_checkpoint(
                    save_path=best_path,
                    decoder=decoder,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_loss=best_loss,
                    avg_loss=avg_loss,
                    cfg=cfg,
                    decoder_cfg_dict=decoder_cfg_dict,
                )

                print(f"Saved best decoder checkpoint: {best_path}")

        if distributed:
            dist.barrier()

    if is_main_process():
        print("\nDecoder training finished.")
        print(f"Last checkpoint: {last_path}")
        print(f"Best checkpoint: {best_path}")

    return decoder


# ============================================================
# Main
# ============================================================

@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def main(cfg):
    """
    从 yaml 读取 LeWM 配置。
    从 decoder_config.py 读取 decoder 配置。

    单卡:
        python decoder_train_ddp.py

    单节点多卡:
        torchrun --nproc_per_node=4 decoder_train_ddp.py

    只用前 7 张卡:
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 decoder_train_ddp.py
    """
    distributed, rank, world_size, local_rank = setup_distributed()

    try:
        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 你原代码里有一个单独写死的 lewm_ckpt，这里保留你的逻辑
        # 如果你希望完全从 decoder_cfg 读取，可以把下面这一行改成:
        # lewm_ckpt = decoder_cfg["paths"]["lewm_ckpt_path"]
        lewm_ckpt = r"C:\Users\wangh\Desktop\world_model\le-wm\cache\20260522_180443\lewm_best.ckpt"

        path_cfg = decoder_cfg["paths"]
        model_cfg = decoder_cfg["model"]
        train_cfg = decoder_cfg["train"]

        if is_main_process():
            print("========== Train Visualization Decoder ==========")
            print(f"distributed   = {distributed}")
            print(f"rank          = {rank}")
            print(f"world_size    = {world_size}")
            print(f"local_rank    = {local_rank}")
            print(f"device        = {device}")
            print(f"lewm_ckpt     = {lewm_ckpt}")
            print(f"h5_path       = {path_cfg['h5_path']}")
            print(f"resume_from   = {path_cfg['resume_from']}")

        # --------------------------------------------------------
        # 1. Dataset
        # --------------------------------------------------------
        train_loader, val_loader = load_train_val(
            cfg,
            path_cfg["h5_path"],
        )

        if distributed:
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler,
                num_workers=getattr(train_loader, "num_workers", 0),
                pin_memory=True,
                drop_last=True,
                persistent_workers=getattr(train_loader, "num_workers", 0) > 0,
            )

        if is_main_process():
            print("Dataset loaded by utils1.load_dataset.load_train_val(cfg)")
            print(f"train_loader batches = {len(train_loader)}")
            print(f"val_loader batches   = {len(val_loader)}")

        # --------------------------------------------------------
        # 2. Build and load frozen LeWM world model
        # --------------------------------------------------------
        world_model = build_world_model(cfg)

        world_model = load_wm_checkpoint(
            model=world_model,
            ckpt_path=lewm_ckpt,
            device=device,
        )

        world_model.to(device)
        world_model.eval()

        for p in world_model.parameters():
            p.requires_grad_(False)

        lewm_encoder = LeWMEncodeWrapper(world_model).to(device)

        # --------------------------------------------------------
        # 3. Build decoder in main
        # --------------------------------------------------------
        decoder = VisualizationDecoder(**model_cfg).to(device)

        # DDP 只包 decoder
        if distributed:
            decoder = DDP(
                decoder,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # --------------------------------------------------------
        # 4. Build DecoderTrainingModel in main
        # --------------------------------------------------------
        model = DecoderTrainingModel(
            lewm_encoder=lewm_encoder,
            decoder=decoder,
        ).to(device)

        model.lewm_encoder.eval()
        model.lewm_encoder.requires_grad_(False)

        # --------------------------------------------------------
        # 5. Build optimizer in main
        # --------------------------------------------------------
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # --------------------------------------------------------
        # 6. Prepare run directory
        # --------------------------------------------------------
        base_run_dir = Path(path_cfg["save_base_dir"])

        # DDP 下所有 rank 必须使用同一个 run_id
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

            # 保存 LeWM yaml 配置
            with open(run_dir / "wm_config.yaml", "w", encoding="utf-8") as f:
                OmegaConf.save(cfg, f)

            # 保存 decoder 字典配置
            torch.save(decoder_cfg, run_dir / "decoder_config.pt")

            # 同时保存一个可读 txt
            with open(run_dir / "decoder_config.txt", "w", encoding="utf-8") as f:
                f.write(repr(decoder_cfg))

            print(f"run_dir = {run_dir}")

        if distributed:
            dist.barrier()

        # --------------------------------------------------------
        # 7. Train decoder
        # --------------------------------------------------------
        train_visualization_decoder(
            model=model,
            decoder=decoder,
            optimizer=optimizer,
            dataloader=train_loader,
            cfg=cfg,
            decoder_cfg_dict=decoder_cfg,
            device=device,
            epochs=train_cfg["epochs"],
            save_dir=run_dir,
            output_name=path_cfg["output_name"],
            resume_from=path_cfg["resume_from"],
            save_every_steps=train_cfg["save_every_steps"],
            distributed=distributed,
        )

        if is_main_process():
            print("Decoder training finished.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
###########################
import torch
from omegaconf import open_dict
from torch.utils.data.distributed import DistributedSampler

from my_swm import HDF5Dataset
from my_spt import Compose, random_split
from utils import get_img_preprocessor, get_column_normalizer


def load_train_val(
    cfg,
    h5_path,
    distributed=False,
    rank=0,
    world_size=1,
):
    """
    构建 train_loader 和 val_loader。

    参数:
        cfg:
            Hydra / OmegaConf 配置
        h5_path:
            HDF5 数据路径
        distributed:
            是否使用 DDP
        rank:
            当前进程 rank
        world_size:
            总进程数

    返回:
        train_loader, val_loader, train_sampler
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
    # 3. DataLoader settings
    # ============================================================
    loader_kwargs = dict(cfg.loader)

    # 防止 cfg.loader 里已有 shuffle，后面重复传参
    loader_kwargs.pop("shuffle", None)

    # 防止 cfg.loader 里已有 drop_last，后面重复传参
    loader_kwargs.pop("drop_last", None)

    # 防止 cfg.loader 里已有 generator，后面重复传参
    loader_kwargs.pop("generator", None)

    # ============================================================
    # 4. Distributed sampler
    # ============================================================
    if distributed:
        #train_sampler是“索引分配器,告诉当前进程：你应该读取哪些数据索引
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True, #表示每个 epoch 重新打乱数据后再切分。
            drop_last=True,  #表示如果数据集不能被 world_size 整除，就丢掉最后少量样本，让每个 rank 拿到相同数量的数据。
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            **loader_kwargs,
            sampler=train_sampler, #DataLoader 不再自己决定怎么取数据，而是听 DistributedSampler 的。
            shuffle=False,
            drop_last=True, #保证每个 rank 内部最后一个 batch 是完整 batch
        )

    else:
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            **loader_kwargs,
            shuffle=True,
            drop_last=True,
            generator=rnd_gen,
        )

    # ============================================================
    # 5. Validation loader
    # ============================================================
    # val 一般不需要 DistributedSampler。
    # 如果只是训练 decoder，通常只用 train_loader 即可。
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **loader_kwargs,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler

train_loader, val_loader, train_sampler = load_train_val(
    cfg,
    path_cfg["h5_path"],
    distributed=distributed,
    rank=rank,
    world_size=world_size,
)