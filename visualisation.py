import torch
from models import SimCLR, BaseEncoder
from models2 import SimCLR2, BaseEncoder2
from torch.utils.data import DataLoader
from data_loader import SuperDARNDataset, contrastive_collate_fn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os
from datetime import datetime
from visualisation_pca import pca_plot_radius

# Create dataset and dataloader
dataset = SuperDARNDataset(
    h5_file_path=r"C:\Users\aman\Desktop\MPhys Data\Data\all1995\test.h5",  
    negative_value=-9999,
    apply_augmentations=False  
)

data_loader = DataLoader(
    dataset,
    batch_size=500,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=contrastive_collate_fn
)

# Load model 
def load_model(path, device="cuda"):
    base_encoder = BaseEncoder(input_channels=1)
    model = SimCLR(base_encoder, projection_dim=128, temperature=0.5, device=device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device)
    model.eval()
    return model

# get embeddings from model
def get_embeddings(model, data_loader, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for batch_data, segment_names, data_1_unscaled, _ in tqdm(
            data_loader, desc="Extracting embeddings", unit="batch"
        ):
            print("Got batch of size:", batch_data.size())
            original_data = torch.stack([torch.from_numpy(arr) for arr in data_1_unscaled], dim=0)
            original_data = original_data.unsqueeze(1).to(device)
            x = batch_data.unsqueeze(1).to(device)
            emb = model.encoder(x, original_data=original_data)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def get_metadata(segment_path, h5_file_path):
    """
    Retrieves the beam_number, start_time, end_time, and mag_lat attributes
    """
    with h5py.File(h5_file_path, 'r') as hf:
        grp = hf[segment_path]  
        beam_number = grp.attrs.get("beam_number", "Unknown")
        start_time  = grp.attrs.get("start_time", "Unknown")
        end_time    = grp.attrs.get("end_time", "Unknown")
        mag_lat_str = grp.attrs.get("mag_lat", None)
        if mag_lat_str is not None:
            mag_lat = [float(x) for x in mag_lat_str.split(",")]
        else:
            mag_lat = None
    return beam_number, start_time, end_time, mag_lat

def parse_flexible_datetime(dt_str):
    """
    Parses a datetime string that may be incomplete.

    """
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        date_part, time_part = dt_str.split("T")
        time_components = time_part.split(":")
        while len(time_components) < 3:
            time_components.append("00")
        fixed_time_str = ":".join(time_components)
        return datetime.fromisoformat(f"{date_part}T{fixed_time_str}")

# visualise anchor and neighbours
def compare_visualize_neighbours(
    embs_no, embs_att, dataset, segment_names,
    k=3,
    anchor_beam=None, anchor_date=None,
    anchor_start_time=None, anchor_end_time=None,
    filter_date=True
):
    """
    Visualize anchor segment and its k nearest neighbours based on embeddings.
    """
    negative_value = -9999
    embs_no_norm = F.normalize(embs_no, p=2, dim=1)
    embs_att_norm = F.normalize(embs_att, p=2, dim=1)
    N = embs_no.shape[0]

    def sample_meets_criteria(seg_path):
        beam, start_time, end_time, _ = get_metadata(seg_path, dataset.h5_file_path)
        meets_beam = (str(beam) == str(anchor_beam)) if anchor_beam is not None else True
        meets_date = True
        meets_start = True
        meets_end = True
        if anchor_date is not None:
            try:
                sample_date = parse_flexible_datetime(start_time).date()
                anchor_date_obj = (parse_flexible_datetime(anchor_date).date()
                                   if isinstance(anchor_date, str) else anchor_date)
                meets_date = (sample_date == anchor_date_obj)
            except Exception:
                meets_date = False
        if anchor_start_time is not None:
            try:
                sample_start_dt = parse_flexible_datetime(start_time)
                anchor_start_dt = (parse_flexible_datetime(anchor_start_time)
                                    if isinstance(anchor_start_time, str) else anchor_start_time)
                meets_start = (sample_start_dt.hour == anchor_start_dt.hour)
            except Exception:
                meets_start = False
        if anchor_end_time is not None:
            try:
                sample_end_dt = parse_flexible_datetime(end_time)
                anchor_end_dt = (parse_flexible_datetime(anchor_end_time)
                                  if isinstance(anchor_end_time, str) else anchor_end_time)
                meets_end = (sample_end_dt.hour == anchor_end_dt.hour)
            except Exception:
                meets_end = False
        return meets_beam and meets_date and meets_start and meets_end

    # select anchor index
    matching_indices = [i for i, seg in enumerate(segment_names) if sample_meets_criteria(seg)]
    if not matching_indices:
        print("No anchor sample found matching criteria. Using random anchor.")
        anchor_idx = np.random.randint(N)
    else:
        anchor_idx = matching_indices[0]

    # parse anchor metadata
    anchor_seg = segment_names[anchor_idx]
    _, anchor_start, _, anchor_mag_lat = get_metadata(anchor_seg, dataset.h5_file_path)
    anchor_date_parsed = parse_flexible_datetime(anchor_start).date()

    # function to select neighbours
    def select_neighbours(emb_norm):
        sim = torch.mv(emb_norm, emb_norm[anchor_idx])
        sim[anchor_idx] = float('-inf')
        sorted_idx = torch.argsort(sim, descending=True).tolist()
        if filter_date and anchor_date is not None:
            neighbours = []
            for idx in sorted_idx:
                dt_str = get_metadata(segment_names[idx], dataset.h5_file_path)[1]
                if parse_flexible_datetime(dt_str).date() == anchor_date_parsed:
                    continue
                neighbours.append(idx)
                if len(neighbours) >= k:
                    break
        else:
            neighbours = sorted_idx[:k]
        return neighbours


    no_neigh = select_neighbours(embs_no_norm)
    att_neigh = select_neighbours(embs_att_norm)

    anchor_data, *_ = dataset[anchor_idx]
    anchor_arr = anchor_data.numpy()
    mask = anchor_arr != negative_value
    vmin, vmax = anchor_arr[mask].min(), anchor_arr[mask].max()

    # set extent if mag_lat available
    if anchor_mag_lat:
        extent_anchor = [0, anchor_arr.shape[0], min(anchor_mag_lat), max(anchor_mag_lat)]
    else:
        extent_anchor = None

    cols = 1 + k
    fig, axs = plt.subplots(2, cols, figsize=(5*cols, 10), sharey=True)

    # plot anchor
    for row in (0, 1):
        ax = axs[row, 0]
        im = ax.imshow(
            np.ma.masked_where(anchor_arr==negative_value, anchor_arr).T,
            aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
            extent=extent_anchor
        )
        title = f"Anchor ({'No Attn' if row==0 else 'Attn'}):\n{anchor_start}"
        ax.set_title(title)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Magnetic Latitude')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    def plot_neighbours(neigh_list, row):
        for j, idx in enumerate(neigh_list, start=1):
            data, *_ = dataset[idx]
            arr = data.numpy()
            beam, start, _, mag = get_metadata(segment_names[idx], dataset.h5_file_path)
            ext = [0, arr.shape[0], min(mag), max(mag)] if mag else None
            ax = axs[row, j]
            im = ax.imshow(
                np.ma.masked_where(arr==negative_value, arr).T,
                aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                extent=ext
            )
            ax.set_title(f"{'No Attn' if row==0 else 'Attn'} Nbr {j}:\n{start}")
            ax.set_xlabel('Time Steps')
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plot_neighbours(no_neigh, 0)
    plot_neighbours(att_neigh, 1)

    plt.tight_layout()
    plt.show()

    return anchor_idx, no_neigh, att_neigh

# visualize PCA with radius and DBSCAN
def compare_visualize_neighbours_paginated(
    embs_no, embs_att, dataset, segment_names,
    k=30,
    anchor_beam=None, anchor_date=None,
    anchor_start_time=None, anchor_end_time=None,
    filter_date=True,
    per_page=10
):
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    negative_value = -9999
    embs_no_norm = F.normalize(embs_no, p=2, dim=1)
    embs_att_norm = F.normalize(embs_att, p=2, dim=1)
    N = embs_no.shape[0]

    def sample_meets_criteria(seg_path):
        beam, start_time, end_time, _ = get_metadata(seg_path, dataset.h5_file_path)
        date_match = anchor_date is None or parse_flexible_datetime(start_time).date() == parse_flexible_datetime(anchor_date).date()
        beam_match = anchor_beam is None or str(beam) == str(anchor_beam)
        return date_match and beam_match

    matching_indices = [i for i, seg in enumerate(segment_names) if sample_meets_criteria(seg)]
    anchor_idx = matching_indices[0] if matching_indices else np.random.randint(N)
    anchor_seg = segment_names[anchor_idx]
    _, anchor_start, _, anchor_mag_lat = get_metadata(anchor_seg, dataset.h5_file_path)
    anchor_date_parsed = parse_flexible_datetime(anchor_start).date()

    def select_neighbours(emb_norm):
        sim = torch.mv(emb_norm, emb_norm[anchor_idx])
        sim[anchor_idx] = float('-inf')
        sorted_idx = torch.argsort(sim, descending=True).tolist()
        if filter_date and anchor_date is not None:
            return [idx for idx in sorted_idx if parse_flexible_datetime(get_metadata(segment_names[idx], dataset.h5_file_path)[1]).date() != anchor_date_parsed][:k]
        return sorted_idx[:k]

    no_neigh = select_neighbours(embs_no_norm)
    att_neigh = select_neighbours(embs_att_norm)

    anchor_data, *_ = dataset[anchor_idx]
    anchor_arr = anchor_data.numpy()
    mask = anchor_arr != negative_value
    vmin, vmax = anchor_arr[mask].min(), anchor_arr[mask].max()

    num_pages = ceil(k / per_page)

    for page in range(num_pages):
        fig, axs = plt.subplots(2, per_page + 1, figsize=(3 * (per_page + 1), 6), sharey=True)
        fig.suptitle(f'Page {page + 1}/{num_pages} — Anchor + Neighbours')

        for row in (0, 1):
            ax = axs[row, 0]
            im = ax.imshow(np.ma.masked_where(anchor_arr == negative_value, anchor_arr).T,
                   aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            beam_anchor, *_ = get_metadata(anchor_seg, dataset.h5_file_path)
            ax.set_title(f"Anchor | Beam {beam_anchor}\n{anchor_start}", fontsize=8)
            ax.set_xlabel('Time')
            ax.set_ylabel('Lat')
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.02)


        def plot_neighbours(row, neighbours):
            start_idx = page * per_page
            end_idx = min((page + 1) * per_page, k)
            for col, i in enumerate(range(start_idx, end_idx), start=1):
                idx = neighbours[i]
                data, *_ = dataset[idx]
                arr = data.numpy()
                beam, start_time, _, mag = get_metadata(segment_names[idx], dataset.h5_file_path)
                ext = [0, arr.shape[0], min(mag), max(mag)] if mag else None
                ax = axs[row, col]
                im = ax.imshow(np.ma.masked_where(arr == negative_value, arr).T,
                       aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=ext)
                ax.set_title(f"Nbr {i+1} | Beam {beam}\n{start_time}", fontsize=7)
                ax.set_xlabel('Time')
                fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.02)


        plot_neighbours(0, no_neigh)
        plot_neighbours(1, att_neigh)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

def get_or_compute_embeddings(model, data_loader, device="cuda", save_path="embeddings.pt", load_mode=True):
    if load_mode and os.path.exists(save_path):
        print(f"Loading embeddings from {save_path}")
        embeddings = torch.load(save_path)
    else:
        print("Computing embeddings...")
        embeddings = get_embeddings(model, data_loader, device)
        torch.save(embeddings, save_path)
        print(f"Saved embeddings to {save_path}")
    return embeddings

# main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path_att = r"C:\Users\aman\OneDrive - University of Southampton\Desktop\Year 4\MPhys Project\Lo\Masters-Project\aman\SimCLR\best_model_attention.pth"
    model_att = load_model(model_path_att, device)
    embeddings_save_path_att = r"C:\Users\aman\OneDrive - University of Southampton\Desktop\Year 4\MPhys Project\Lo\Masters-Project\aman\SimCLR\embeddings_attention.pt"
    embeddings_att = get_or_compute_embeddings(model_att, data_loader, device, save_path=embeddings_save_path_att, load_mode=True)

    print("Embeddings shape (no attention):", embeddings_no.shape)
    print("Embeddings shape (attention):", embeddings_att.shape)
    segment_names = dataset.segments
    embeddings_no = F.normalize(embeddings_no, p=2, dim=1)
    embeddings_att = F.normalize(embeddings_att, p=2, dim=1)
    compare_visualize_neighbours(embeddings_no, embeddings_att, dataset, segment_names, k=4, 
                                  anchor_beam=0, 
                                  anchor_date="1995-09-24",
                                  anchor_start_time="1995-09-24T07", 
                                  anchor_end_time=None,
                                  filter_date=False)
    
    pca_plot_radius(
        embeddings_no,   
        embeddings_att,
        dataset,
        segment_names,
        embeddings_type="att",
        #radius=0.02,
        top_k = 50,
        anchor_beam=0,
        anchor_date="1995-09-24",
        anchor_start_time="1995-09-24T07",   
        anchor_end_time="1995-09-24T08",
        output_txt_path="neighbours_output.txt"  
        )
    
    compare_visualize_neighbours_paginated(
        embeddings_no, embeddings_att, dataset, segment_names,
        k=50, anchor_beam=0, anchor_date="1995-09-24",anchor_end_time="1995-09-24T08", anchor_start_time="1995-09-24T07",
        filter_date=False, per_page=10
    )
if __name__ == "__main__":
    main()
