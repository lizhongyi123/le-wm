import torch
def unnormalize_imagenet(x):
    """
    x: (..., 3, H, W)
    支持:
        (T, 3, H, W)
        (N, 3, H, W)
        (B, T, 3, H, W)
    """
    device = x.device
    dtype = x.dtype

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
        dtype=dtype,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
        dtype=dtype,
    )

    # 让 mean/std 自动匹配 x 的维度
    shape = [1] * x.ndim
    shape[-3] = 3

    mean = mean.view(*shape)
    std = std.view(*shape)

    x = x * std + mean
    return x.clamp(0.0, 1.0)