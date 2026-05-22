# decoder_train.py
from pathlib import Path
import torch
from tqdm import tqdm
from omegaconf import open_dict

from utils import get_column_normalizer, get_img_preprocessor
from my_swm import HDF5Dataset
from my_spt import Compose, vit_hf

from decoder_model import VisualizationDecoder, DecoderTrainingModel


def build_decoder_dataset(cfg, h5_data_path):
    dataset = HDF5Dataset(
        **cfg.data.dataset,
        transform=None,
        h5_data_path=h5_data_path,
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

    dataset.transform = Compose(*transforms)

    return dataset


def build_encoder_from_cfg(cfg):
    return vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )


def load_encoder_from_lewm_checkpoint(encoder, ckpt_path, device="cuda"):
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    if not encoder_state:
        raise KeyError(
            f"No encoder weights found in checkpoint: {ckpt_path}"
        )

    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device)
    encoder.eval()

    print(f"Loaded LeWM encoder checkpoint: {ckpt_path}")
    print(f"epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"global_step: {ckpt.get('global_step', 'unknown')}")

    return encoder


def move_pixels_to_device(batch, device):
    if isinstance(batch, dict):
        pixels = batch["pixels"]
    elif isinstance(batch, (tuple, list)):
        pixels = batch[0]
    else:
        pixels = batch

    if pixels.ndim == 5:
        B, T, C, H, W = pixels.shape
        pixels = pixels.reshape(B * T, C, H, W)

    return pixels.to(device).float()


def save_decoder_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    global_step,
    best_loss,
    avg_loss,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "avg_loss": avg_loss,
        },
        save_path,
    )


def load_decoder_checkpoint(
    ckpt_path,
    model,
    optimizer=None,
    device="cuda",
):
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model.decoder.load_state_dict(ckpt["decoder"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))

    print(f"Loaded decoder checkpoint: {ckpt_path}")
    print(f"Resume from epoch: {start_epoch}")
    print(f"Global step: {global_step}")
    print(f"Best loss: {best_loss}")

    return start_epoch, global_step, best_loss


def train_visualization_decoder(
    encoder,
    dataloader,
    device="cuda",
    epochs=20,
    lr=1e-4,
    save_dir="decoder_ckpts",
    output_name="visualization_decoder",
    resume_from=None,
    cls_dim=192,
    hidden_dim=256,
    image_size=224,
    patch_size=16,
    channels=3,
    depth=4,
    heads=8,
    dim_head=64,
):
    """
    Train visualization decoder.

    如果 resume_from 不为空，则从该 checkpoint 继续训练。
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / f"{output_name}_last.pt"
    best_path = save_dir / f"{output_name}_best.pt"

    decoder = VisualizationDecoder(
        cls_dim=cls_dim,
        hidden_dim=hidden_dim,
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
    )

    model = DecoderTrainingModel(
        encoder=encoder,
        decoder=decoder,
        freeze_encoder=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if resume_from is not None and str(resume_from).lower() not in ["", "none", "null"]:
        start_epoch, global_step, best_loss = load_decoder_checkpoint(
            ckpt_path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    model.train()
    model.encoder.eval()

    if start_epoch >= epochs:
        print(f"start_epoch={start_epoch} >= epochs={epochs}, no training needed.")
        return model.decoder

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Decoder Train Epoch {epoch + 1}/{epochs}",
            dynamic_ncols=True,
        )

        for batch in pbar:
            pixels = move_pixels_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            output = model.compute_loss(pixels)
            loss = output["loss"]

            loss.backward()
            optimizer.step()

            global_step += 1

            loss_value = output["loss"].item()
            l1_value = output["loss_l1"].item()
            l2_value = output["loss_l2"].item()

            total_loss += loss_value
            total_l1 += l1_value
            total_l2 += l2_value

            pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}",
                    "l1": f"{l1_value:.6f}",
                    "l2": f"{l2_value:.6f}",
                }
            )

        n = max(len(dataloader), 1)
        avg_loss = total_loss / n
        avg_l1 = total_l1 / n
        avg_l2 = total_l2 / n

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]\n"
            f"  loss = {avg_loss:.6f}\n"
            f"  l1   = {avg_l1:.6f}\n"
            f"  l2   = {avg_l2:.6f}\n"
        )

        save_decoder_checkpoint(
            save_path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_loss=best_loss,
            avg_loss=avg_loss,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            save_decoder_checkpoint(
                save_path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_loss=best_loss,
                avg_loss=avg_loss,
            )

            print(f"Saved best decoder checkpoint: {best_path}")

    print("Decoder training finished.")
    print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {best_path}")

    return model.decoder
