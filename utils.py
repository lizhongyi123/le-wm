import numpy as np
import torch
from pathlib import Path
# from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from my_spt import ToImage, Resize, Compose,WrapTorchTransform

import dataset_stats

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dataset_stats.ImageNet
    to_image = ToImage(**imagenet_stats, source=source, target=target)
    resize = Resize(img_size, source=source, target=target)
    return Compose(to_image, resize)


#根据数据集某一列的数据，计算这一列的均值和标准差，然后构造一个归一化 transform，用于把该列数据标准化为均值约为 0、标准差约为 1 的数据。
# def get_column_normalizer(dataset, source: str, target: str):
#     """Get normalizer for a specific column in the dataset."""
#     col_data = dataset.get_col_data(source)
#     data = torch.from_numpy(np.array(col_data))
#     data = data[~torch.isnan(data).any(dim=1)]
#     mean = data.mean(0, keepdim=True).clone()
#     std = data.std(0, keepdim=True).clone()
#
#     def norm_fn(x):
#         return ((x - mean) / std).float()
#
#     normalizer = WrapTorchTransform(norm_fn, source=source, target=target)
#     return normalizer

class ColumnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]

    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    std = torch.clamp(std, min=1e-6)

    return WrapTorchTransform(
        ColumnNormalize(mean, std),
        source=source,
        target=target
    )

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")