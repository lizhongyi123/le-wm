from torchvision.transforms import AutoAugmentPolicy, InterpolationMode  # usort: skip
from torchvision.transforms import v2
import itertools
import math
import warnings
from collections.abc import Sequence
from typing import Optional, Union, cast


from torch import Generator, default_generator, randperm
from torch import nn
import torch

from torchvision import tv_tensors
import PIL.Image
import numpy as np
import torchvision
import logging
from transformers import ViTConfig, ViTModel


class Compose(v2.Transform):
    """Compose multiple transforms together in sequence."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def __call__(self, sample):
        for a in self.args:
            sample = a(sample)
        return sample

class Dataset(torch.utils.data.Dataset):
    """Base dataset class with transform support and PyTorch Lightning integration."""

    def __init__(self, transform=None):
        self.transform = transform
        self._trainer = None

    def set_pl_trainer(self, trainer):
        self._trainer = trainer

    def process_sample(self, sample, **kwargs):
        for k, v in kwargs.items():
            sample[k] = v
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class Subset(Dataset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    @property
    def column_names(self):
        return self.dataset.column_names

def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> list[Subset]:
    r"""Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)

        #处理剩余数据
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]

def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Vision Transformer using HuggingFace transformers.

    This provides a clean, well-maintained ViT implementation with native support for:
    - Masking via bool_masked_pos parameter
    - Learnable mask token
    - Easy access to CLS and patch tokens

    Args:
        size: Model size - "tiny", "small", "base", or "large"
        patch_size: Patch size (default: 16)
        image_size: Input image size (default: 224)
        pretrained: Load pretrained weights from HuggingFace Hub
        use_mask_token: Whether to include learnable mask token (needed for iBOT)
        **kwargs: Additional ViTConfig parameters

    Returns:
        HuggingFace ViTModel

    Example:
        >>> backbone = vit_hf("tiny", use_mask_token=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>>
        >>> # Without masking
        >>> output = backbone(x)
        >>> cls_token = output.last_hidden_state[:, 0, :]
        >>> patch_tokens = output.last_hidden_state[:, 1:, :]
        >>>
        >>> # With masking (for iBOT student)
        >>> masks = torch.zeros(2, 196, dtype=torch.bool)
        >>> masks[:, :59] = True  # Mask 30%
        >>> output = backbone(x, bool_masked_pos=masks)
    """

    # ViT size configurations (matching timm/DINOv3)
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
        "huge": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
        },
    }

    if size not in size_configs:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(size_configs.keys())}"
        )

    config_params = size_configs[size]
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        # Try to load pretrained model from HF Hub
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        logging.info(f"Loading pretrained ViT from {model_name}")
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        logging.info(f"Created ViT-{size} from scratch with config: {config_params}")

    # IMPORTANT: Set model to always interpolate position encodings for dynamic input sizes
    # This allows processing images of different sizes (e.g., 224x224 global + 96x96 local views)
    # Must be set as instance attribute, not in config
    model.config.interpolate_pos_encoding = True

    return model

class Transform(v2.Transform):
    """Base transform class extending torchvision v2.Transform with nested data handling."""

    def single_nested_get(self, v, name):
        if name == "":
            return v
        i = name.split(".")
        if i[0].isnumeric():
            i[0] = int(i[0])
        return self.single_nested_get(v[i[0]], ".".join(i[1:]))

    def nested_get(self, v, name):
        if type(name) in [list, tuple]:
            return [self.single_nested_get(v, n) for n in name]
        return self.single_nested_get(v, name)

    def single_nested_set(self, original, value, name):
        if "." not in name:
            if name.isnumeric():
                name = int(name)
            original[name] = value
        else:
            i = name.split(".")
            if i[0].isnumeric():
                i[0] = int(i[0])
            self.single_nested_set(original[i[0]], value, ".".join(i[1:]))

    def nested_set(self, original, value, name):
        if type(name) in [list, tuple]:
            assert type(value) in [list, tuple]
            assert len(value) == len(name)
            return [self.single_nested_set(original, v, n) for v, n in zip(value, name)]
        return self.single_nested_set(original, value, name)

    def get_name(self, x):
        base = self.name
        assert "_" not in base
        if base not in x:
            return base
        ctr = 0
        while f"{base}_{ctr}" in base:
            ctr += 1
        return f"{base}_{ctr}"

    @property
    def name(self):
        return self.__class__.__name__

@torch.jit.unused
def to_image(
    input: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(input, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(input)).transpose(-3, -1).contiguous()
    elif isinstance(input, PIL.Image.Image):
        output = torchvision.transforms.functional.pil_to_tensor(input)
    elif isinstance(input, torch.Tensor):
        output = input
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(input)} instead."
        )
    return tv_tensors.Image(output)

class ToImage(Transform):
    """Convert input to image tensor with optional normalization."""

    def __init__(
        self,
        dtype=torch.float32,
        scale=True,
        mean=None,
        std=None,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__()
        t = [to_image, v2.ToDtype(dtype, scale=scale)]
        if mean is not None and std is not None:
            t.append(v2.Normalize(mean=mean, std=std))
        self.t = v2.Compose(t)
        self.source = source
        self.target = target

    def __call__(self, x):
        #x["image"]
        self.nested_set(x, self.t(self.nested_get(x, self.source)), self.target)
        return x

class Resize(Transform, v2.Resize):
    """Resize image to specified size."""

    def __init__(
        self,
        size,
        interpolation=2,
        max_size=None,
        antialias=True,
        source="image",
        target="image",
    ) -> None:
        super().__init__(size, interpolation, max_size, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self._transform(self.nested_get(x, self.source), []), self.target
        )
        return x

class WrapTorchTransform(Transform, v2.Lambda):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, transform, source: str = "image", target: str = "image"):
        super().__init__(transform)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, super().__call__(self.nested_get(x, self.source)), self.target
        )
        return x