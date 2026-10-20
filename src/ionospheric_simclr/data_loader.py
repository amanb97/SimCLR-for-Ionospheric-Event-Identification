"""HDF5 power segments and the original SimCLR batch ordering."""
from pathlib import Path
import re
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentation_strategies import augment_power


def segment_paths(path):
    """Accept segment/data, beam_N_segment_M/data and beam_N/segment_M/data."""
    names = []
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('/data'):
                names.append(name.rsplit('/', 1)[0])
        f.visititems(visit)
    return sorted(names, key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)])


class SuperDARNDataset(Dataset):
    # Original statistics, used only if not replaced by a new run/checkpoint.
    global_power_mean = 15.023935
    global_power_std = 9.644889

    def __init__(self, h5_file_path, negative_value=-9999, apply_augmentations=True,
                 augment_params=None, max_samples=None):
        self.h5_file_path = str(Path(h5_file_path).resolve())
        self.negative_value = negative_value
        self.apply_augmentations = apply_augmentations
        self.augment_params = augment_params or {}
        self.segments = segment_paths(self.h5_file_path)
        if max_samples is not None:
            self.segments = self.segments[:max_samples]
        if not self.segments:
            raise ValueError(f'No segment/data datasets found in {h5_file_path}')
        shape = None
        with h5py.File(self.h5_file_path, 'r') as f:
            for name in self.segments:
                current = f[name]['data'].shape
                if len(current) != 3 or current[2] < 1 or min(current[:2]) < 4:
                    raise ValueError(f'{name}: expected (time>=4, spatial_bins>=4, features>=1), got {current}')
                if shape is not None and current[:2] != shape:
                    raise ValueError('Segments must share temporal/spatial dimensions for batching.')
                shape = current[:2]
        self.sample_shape = shape

    def __len__(self):
        return len(self.segments)

    def raw_power(self, idx):
        with h5py.File(self.h5_file_path, 'r') as f:
            power = np.asarray(f[self.segments[idx]]['data'][:, :, 0], dtype=np.float32)
        power[~np.isfinite(power)] = self.negative_value
        return power

    def compute_statistics(self):
        count, mean, m2 = 0, 0.0, 0.0
        with h5py.File(self.h5_file_path, 'r') as f:
            for name in self.segments:
                raw = np.asarray(f[name]['data'][:, :, 0], dtype=np.float64)
                values = raw[np.isfinite(raw) & (raw != self.negative_value)]
                if not len(values):
                    raise ValueError(f'{name} contains no valid power observations.')
                n = len(values)
                batch_mean = float(values.mean())
                delta = batch_mean - mean
                m2 += float(((values - batch_mean) ** 2).sum()) + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
        return mean, max(float(np.sqrt(m2 / count)), 1e-6)

    def normalise(self, power):
        scaled = power.copy()
        mask = scaled != self.negative_value
        scaled[mask] = (scaled[mask] - self.global_power_mean) / self.global_power_std
        # Preserve the research encoder's input convention for checkpoint compatibility.
        return torch.from_numpy(scaled).float()

    def __getitem__(self, idx):
        raw = self.raw_power(idx)
        if not np.any(raw != self.negative_value):
            raise ValueError(f'{self.segments[idx]} contains no valid power observations.')
        if not self.apply_augmentations:
            return self.normalise(raw), None, raw, None, self.segments[idx]
        a = augment_power(raw, **self.augment_params)
        b = augment_power(raw, **self.augment_params)
        if not np.any(a != self.negative_value):
            a = raw.copy()
        if not np.any(b != self.negative_value):
            b = raw.copy()
        return self.normalise(a), self.normalise(b), a, b, self.segments[idx]


def contrastive_collate_fn(batch):
    names = [item[4] for item in batch]
    first = torch.stack([item[0] for item in batch])
    raw_first = [item[2] for item in batch]
    if batch[0][1] is None:
        return first, names, raw_first, None
    second = torch.stack([item[1] for item in batch])
    return torch.cat((first, second)), names, raw_first, [item[3] for item in batch]
