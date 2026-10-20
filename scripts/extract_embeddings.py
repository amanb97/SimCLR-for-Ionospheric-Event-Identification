"""Save encoder embeddings with the corresponding segment IDs and metadata."""
import argparse
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ionospheric_simclr.data_loader import SuperDARNDataset, contrastive_collate_fn
from ionospheric_simclr.models import BaseEncoder, SimCLR
from ionospheric_simclr.utils import choose_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/test.h5')
    p.add_argument('--checkpoint', default='runs/ssl/checkpoints/best_model.pth')
    p.add_argument('--out', default='runs/embeddings/test.npz')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.max_samples is not None and args.max_samples < 1):
        p.error('Batch size, threads and max-samples must be positive.')
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {})
    state = checkpoint['model_state_dict']
    projection_dim = state['projection_head.fc2.weight'].shape[0]
    model = SimCLR(BaseEncoder(), projection_dim=projection_dim,
                   temperature=config.get('temperature', 0.5), device=device).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    ds = SuperDARNDataset(args.input, apply_augmentations=False, max_samples=args.max_samples)
    ds.global_power_mean = config.get('power_mean', ds.global_power_mean)
    ds.global_power_std = config.get('power_std', ds.global_power_std)
    if not config:
        print('Checkpoint has no config: using original loader normalisation (15.023935, 9.644889).')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=contrastive_collate_fn)
    features, names = [], []
    with torch.inference_mode():
        for x, batch_names, raw, _ in loader:
            original = torch.from_numpy(np.stack(raw)).unsqueeze(1).to(device)
            encoded = model.encoder(x.unsqueeze(1).to(device), original_data=original)
            features.append(F.normalize(encoded, dim=1).cpu().numpy())
            names.extend(batch_names)
    embeddings = np.concatenate(features)
    if not np.isfinite(embeddings).all():
        raise ValueError('Encoder produced non-finite embeddings.')
    metadata = []
    with h5py.File(ds.h5_file_path, 'r') as f:
        for name in names:
            metadata.append(json.dumps({k: str(v) for k, v in f[name].attrs.items()}))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=embeddings, segment_names=np.asarray(names),
                        metadata=np.asarray(metadata), input_h5=ds.h5_file_path,
                        checkpoint=str(Path(args.checkpoint).resolve()),
                        power_mean=ds.global_power_mean, power_std=ds.global_power_std)
    print(f'Saved {embeddings.shape} encoder embeddings to {output}')


if __name__ == '__main__':
    main()
