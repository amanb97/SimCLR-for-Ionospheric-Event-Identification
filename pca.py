import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle
import h5py
from datetime import datetime
from typing import Optional, Union, Tuple

__all__ = [
    "pca_plot_radius",
    "visualize_segments_with_anchor",
]

# parse datetimes that may lack minutes/seconds
def parse_flexible_datetime(dt_str: str) -> datetime:
    """Parse ISO datetimes that may lack minutes/seconds."""
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        date_part, time_part = dt_str.split("T")
        comps = time_part.split(":")
        while len(comps) < 3:
            comps.append("00")
        return datetime.fromisoformat(f"{date_part}T{':'.join(comps)}")

# get metadata for a segment
def get_metadata(segment_path: str, h5_file_path: str):
    """Return beam, start, end for an HDF5 segment."""
    with h5py.File(h5_file_path, 'r') as hf:
        grp = hf[segment_path]
        return (grp.attrs.get('beam_number', 'Unknown'),
                grp.attrs.get('start_time', 'Unknown'),
                grp.attrs.get('end_time', 'Unknown'))

# #visualise PCA plot by highlighting neighbours around an anchor point
def pca_plot_radius(
    embs_no: torch.Tensor,
    embs_att: torch.Tensor,
    dataset,
    segment_names: list,
    *,
    embeddings_type: str = "no_att",
    top_k: int = 100,
    anchor_beam: Optional[int] = None,
    anchor_date: Optional[Union[str, datetime]] = None,
    anchor_start_time: Optional[Union[str, datetime]] = None,
    anchor_end_time: Optional[Union[str, datetime]] = None,
    figsize: Tuple[int, int] = (8, 6),
    output_txt_path: str = "neighbour_times.txt",
):
    if embeddings_type not in {'no_att', 'att'}:
        raise ValueError("embeddings_type must be 'no_att' or 'att'")
    embs = embs_no if embeddings_type == 'no_att' else embs_att
    data = embs.detach().cpu().numpy()
    pcs = PCA(n_components=2).fit_transform(data)

    def matches(i):
        beam, start, end = get_metadata(segment_names[i], dataset.h5_file_path)
        if anchor_beam is not None and str(beam) != str(anchor_beam): return False
        if anchor_date is not None:
            sd = parse_flexible_datetime(start).date()
            ad = parse_flexible_datetime(anchor_date).date() if isinstance(anchor_date, str) else anchor_date.date()
            if sd != ad: return False
        if anchor_start_time and parse_flexible_datetime(start).hour != parse_flexible_datetime(anchor_start_time).hour:
            return False
        if anchor_end_time and parse_flexible_datetime(end).hour != parse_flexible_datetime(anchor_end_time).hour:
            return False
        return True

    candidates = [i for i in range(len(segment_names)) if matches(i)]
    anchor_idx = candidates[0] if candidates else np.random.randint(len(segment_names))
    if not candidates:
        print(f"[pca_plot_radius] no anchor matched; using random {anchor_idx}")

    anchor_pt = pcs[anchor_idx]
    dists = np.linalg.norm(pcs - anchor_pt, axis=1)
    sorted_indices = np.argsort(dists)
    topk_indices = sorted_indices[1:top_k + 1]  # exclude anchor

    labels = DBSCAN(eps=0.4, min_samples=20).fit_predict(data)
    cluster1_mask = (labels == 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.3, label='All Points')
    ax.scatter(pcs[topk_indices, 0], pcs[topk_indices, 1], c='C1', label=f'Top-{top_k} Neighbours')
    ax.scatter(*anchor_pt, marker='*', s=200, c='red', label='Anchor')

    circle = Circle(anchor_pt, np.linalg.norm(pcs[topk_indices[-1]] - anchor_pt), facecolor='none', edgecolor='red', ls='--')
    ax.add_patch(circle)

    ax.scatter(pcs[cluster1_mask, 0], pcs[cluster1_mask, 1], marker='x', c='green', s=70, label='DBSCAN Cluster 1')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA ({embeddings_type}) – Top-{top_k} Neighbours, Cluster Overlay')
    ax.legend()
    plt.tight_layout()
    plt.show()

    with open(output_txt_path, "w") as f:
        f.write(f"Anchor: {segment_names[anchor_idx]}\n")
        f.write(f"--- Top-{top_k} Neighbours ---\n")
        for i in topk_indices:
            beam, start, end = get_metadata(segment_names[i], dataset.h5_file_path)
            in_cl1 = labels[i] == 1
            f.write(f"{segment_names[i]} | {start} - {end} | Cluster 1: {in_cl1}\n")


