import os
from pathlib import Path

import h5py
import numpy as np
import torch
from collections.abc import Callable
import os
import logging
import traceback
import hdf5plugin
# ============================================================
# swm.data.utils.get_cache_dir 替代
# ============================================================
import re
def get_cache_dir():
    """
    替代 swm.data.utils.get_cache_dir。

    默认保存到 ./cache。
    也可以通过环境变量 LEWM_CACHE_DIR 指定。
    """
    path = Path(os.environ.get("LEWM_CACHE_DIR", "./cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class _UtilsNamespace:
    get_cache_dir = staticmethod(get_cache_dir)

class Dataset:
    """Base class for episode-based datasets.

    Args:
        lengths: Array of episode lengths.
        offsets: Array of episode start offsets in the data.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample.
        transform: Optional transform to apply to loaded data.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.lengths = lengths
        self.offsets = offsets
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.frameskip, -1
                )
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        """Load full episode by index."""
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError

# ============================================================
# swm.data.HDF5Dataset 替代
# ============================================================

class HDF5Dataset(Dataset):
    """Dataset loading from HDF5 file.

    Reads data from a single .h5 file containing all episode data.
    Uses SWMR mode for robust reading while writing.

    Args:
        name: Name of the dataset (filename without extension).
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional data transform callable.
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        cache_dir: Directory containing the dataset file.
    """

    def __init__(
        self,
        num_steps: int = 1,
        frameskip: int = 1,
        name: str = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        transform: Callable[[dict], dict] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        h5_data_path: str | Path | None = None,
    ) -> None:

        # self.h5_path = Path(cache_dir or get_cache_dir(), f'{name}.h5')
        self.h5_path = os.path.join(h5_data_path, f'{name}.h5')

        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        #每个episode的长度。每个 episode 在总数据中的起始位置
        with h5py.File(self.h5_path, 'r') as f:
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )


    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        # for col in self._keys:
        #     src = self._cache if col in self._cache else self.h5_file
        #     try:
        #         print(
        #             f"[PID {os.getpid()}] ep_idx={ep_idx}, start={start}, end={end}, "
        #             f"g_start={g_start}, g_end={g_end}, col={col}, "
        #             f"cached={col in self._cache}",
        #             flush=True
        #         )
        #         data = src[col][g_start:g_end]
        #     except Exception as e:
        #         print("=" * 80, flush=True)
        #         print(f"[PID {os.getpid()}] HDF5 read failed", flush=True)
        #         print(f"col      = {col}", flush=True)
        #         print(f"ep_idx   = {ep_idx}", flush=True)
        #         print(f"start    = {start}", flush=True)
        #         print(f"end      = {end}", flush=True)
        #         print(f"g_start  = {g_start}", flush=True)
        #         print(f"g_end    = {g_end}", flush=True)
        #         print(f"h5_path  = {self.h5_path}", flush=True)
        #         print(f"cached   = {col in self._cache}", flush=True)
        #         traceback.print_exc()
        #         print("=" * 80, flush=True)
        #         raise

        for col in self._keys:
            #判断从缓存读还是从h5文件读

            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != 'action':
                data = data[:: self.frameskip]

            #[b"push the T block"] 变成 "push the T block"
            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                #(T, H, W, C) 变成 (T, C, H, W)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        self._open()

        if isinstance(source, str):
            source = [k for k in self.h5_file.keys() if re.match(source, k)]

        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1



class _DataNamespace:
    HDF5Dataset = HDF5Dataset
    utils = _UtilsNamespace


data = _DataNamespace