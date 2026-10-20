"""Cluster encoder embeddings, plot PCA and export segment assignments."""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ionospheric_simclr.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embeddings', default='runs/embeddings/test.npz')
    p.add_argument('--out', default='runs/clusters')
    p.add_argument('--eps', type=float, default=0.4)
    p.add_argument('--min-samples', type=int, default=20)
    p.add_argument('--input-h5', help='Optional HDF5 to plot cluster members; defaults to path saved during extraction.')
    args = p.parse_args()
    if args.eps <= 0 or args.min_samples < 2:
        p.error('Use eps > 0 and min-samples >= 2.')
    with np.load(args.embeddings, allow_pickle=False) as f:
        x = f['embeddings']
        names = f['segment_names']
        metadata = f['metadata']
        h5_path = args.input_h5 or str(f['input_h5'])
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite embedding vectors.')
    if len(names) != len(x):
        raise ValueError('Embedding and segment counts do not match.')
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(x)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(*pcs[mask].T, s=18, alpha=0.75,
                   label='Noise' if label == -1 else f'Cluster {label}',
                   color='lightgray' if label == -1 else None)
    ax.set(xlabel='PC1', ylabel='PC2', title='DBSCAN on encoder embeddings — PCA display')
    if len(set(labels)) <= 15:
        ax.legend()
    fig.savefig(out / 'pca_clusters.png', dpi=160)
    plt.close(fig)
    with (out / 'assignments.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['segment', 'cluster', 'pc1', 'pc2', 'metadata_json'])
        writer.writerows((str(n), int(label), float(pc[0]), float(pc[1]), str(meta))
                         for n, label, pc, meta in zip(names, labels, pcs, metadata))
    valid = labels != -1
    n_clusters = len(set(labels[valid]))
    score = None
    if 2 <= n_clusters < int(valid.sum()):
        # Bound silhouette computation on large datasets; sample can omit small clusters.
        try:
            score = float(silhouette_score(x[valid], labels[valid], sample_size=min(2000, int(valid.sum())), random_state=42))
        except ValueError:
            pass
    summary = {'segments': len(x), 'embedding_dimensions': x.shape[1],
               'eps': args.eps, 'min_samples': args.min_samples, 'clusters': n_clusters,
               'noise_segments': int((~valid).sum()), 'silhouette_excluding_noise': score,
               'silhouette_max_samples': 2000, 'distance': 'Euclidean on normalised encoder embeddings',
               'interpretation': 'Cluster IDs are not physical event labels; PCA is for display only.'}
    save_json(out / 'summary.json', summary)
    if Path(h5_path).is_file():
        # Show up to three examples from each of the four largest non-noise clusters.
        ranked = sorted(set(labels[valid]), key=lambda c: int((labels == c).sum()), reverse=True)[:4]
        if not ranked:
            ranked = [-1]
        fig, axes = plt.subplots(len(ranked), 3, squeeze=False, figsize=(12, 3.5 * len(ranked)), constrained_layout=True)
        with h5py.File(h5_path, 'r') as f:
            for row, label in enumerate(ranked):
                members = np.flatnonzero(labels == label)[:3]
                for col, ax in enumerate(axes[row]):
                    if col >= len(members):
                        ax.set_visible(False)
                        continue
                    name = str(names[members[col]])
                    raw = np.asarray(f[name]['data'][:, :, 0])
                    mask = (raw == -9999) | ~np.isfinite(raw)
                    im = ax.imshow(np.ma.masked_where(mask, raw).T, origin='lower', aspect='auto')
                    ax.set(title=f'{"Noise" if label == -1 else f"Cluster {label}"}: {name}',
                           xlabel='Time index', ylabel='Spatial-bin index')
                    fig.colorbar(im, ax=ax, label='Power (dB)')
        fig.savefig(out / 'cluster_examples.png', dpi=160)
        plt.close(fig)
    print(summary)
    print(f'Saved clustering outputs to {out}')


if __name__ == '__main__':
    main()
