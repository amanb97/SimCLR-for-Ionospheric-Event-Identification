"""Inspect processed HDF5 arrays and save example power plots."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ionospheric_simclr.data_loader import SuperDARNDataset
from ionospheric_simclr.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/train.h5')
    p.add_argument('--out', default='runs/inspection')
    p.add_argument('--max-samples', type=int, default=None)
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error('--max-samples must be positive.')
    ds = SuperDARNDataset(args.input, apply_augmentations=False, max_samples=args.max_samples)
    mean, std = ds.compute_statistics()
    missing = total = 0
    lo, hi = np.inf, -np.inf
    for i in range(len(ds)):
        raw = ds.raw_power(i)
        valid = raw != ds.negative_value
        missing += int((~valid).sum())
        total += raw.size
        lo, hi = min(lo, float(raw[valid].min())), max(hi, float(raw[valid].max()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = min(6, len(ds))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, ax in zip(np.linspace(0, len(ds) - 1, n, dtype=int), axes.flat):
        ax.set_visible(True)
        raw = ds.raw_power(int(i))
        im = ax.imshow(np.ma.masked_equal(raw, ds.negative_value).T,
                       origin='lower', aspect='auto', vmin=lo, vmax=hi)
        ax.set(title=ds.segments[int(i)], xlabel='Time index', ylabel='Spatial-bin index')
        fig.colorbar(im, ax=ax, label='Power (dB)')
    fig.suptitle('Processed SuperDARN segments')
    fig.savefig(out / 'samples.png', dpi=160)
    plt.close(fig)
    summary = {'input': ds.h5_file_path, 'segments_inspected': len(ds),
               'power_shape': list(ds.sample_shape), 'power_mean': mean, 'power_std': std,
               'missing_fraction': missing / total, 'valid_min': lo, 'valid_max': hi,
               'note': 'Inspection statistics describe this input; training fits its own statistics.'}
    save_json(out / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
