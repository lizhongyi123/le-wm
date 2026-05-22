# decoder_infer.py
from pathlib import Path
import torch

from decoder_model import VisualizationDecoder, extract_cls_from_encoder_output


@torch.no_grad()
def extract_cls_from_encoder(encoder, pixels):
    return extract_cls_from_encoder_output(encoder(pixels))


def load_visualization_decoder(
    ckpt_path,
    device="cuda",
    cls_dim=192,
    hidden_dim=256,
    image_size=224,
    patch_size=16,
    channels=3,
    depth=4,
    heads=8,
    dim_head=64,
):
    decoder = VisualizationDecoder(
        cls_dim=cls_dim,
        hidden_dim=hidden_dim,
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    else:
        decoder.load_state_dict(ckpt)

    decoder.eval()

    print(f"Loaded visualization decoder: {ckpt_path}")
    return decoder


@torch.no_grad()
def reconstruct_with_decoder(
    encoder,
    decoder,
    pixels,
    device="cuda",
):
    """
    pixels:
        (B, 3, 224, 224)
        or (B, T, 3, 224, 224)

    return:
        recon: same image batch flattened if input is 5D
    """
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    pixels = pixels.to(device).float()

    original_shape = pixels.shape

    if pixels.ndim == 5:
        B, T, C, H, W = pixels.shape
        pixels = pixels.reshape(B * T, C, H, W)

    cls_emb = extract_cls_from_encoder(encoder, pixels)
    recon = decoder(cls_emb)

    if len(original_shape) == 5:
        recon = recon.reshape(B, T, *recon.shape[1:])

    return recon


def save_reconstruction_result(path, pixels, recon):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "pixels": pixels.detach().cpu(),
            "recon": recon.detach().cpu(),
        },
        path,
    )

    print(f"Saved reconstruction result to: {path}")
