"""Inspect HDF5 data, train SimCLR, extract test embeddings and cluster."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='runs/workflow')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--eps', type=float, default=0.4)
    p.add_argument('--min-samples', type=int, default=20)
    args = p.parse_args()
    data, out = Path(args.data_dir).resolve(), Path(args.out_dir).resolve()
    for split in ['train', 'val', 'test']:
        if not (data / f'{split}.h5').is_file():
            p.error(f'Missing {data / (split + ".h5")}. Download data or set --data-dir.')
    if out.exists() and any(out.iterdir()):
        p.error('--out-dir must be new or empty; choose another run name.')
    scripts = Path(__file__).resolve().parent
    limit = ['--max-samples', str(args.max_samples)] if args.max_samples is not None else []
    commands = [
        ['inspect_data.py', '--input', str(data / 'train.h5'), '--out', str(out / 'inspection'), *limit],
        ['train_ssl.py', '--train', str(data / 'train.h5'), '--val', str(data / 'val.h5'),
         '--out', str(out / 'ssl'), '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
         '--device', args.device, *limit],
        ['extract_embeddings.py', '--input', str(data / 'test.h5'),
         '--checkpoint', str(out / 'ssl/checkpoints/best_model.pth'),
         '--out', str(out / 'embeddings/test.npz'), '--batch-size', str(args.batch_size),
         '--device', args.device, *limit],
        ['cluster_dbscan.py', '--embeddings', str(out / 'embeddings/test.npz'),
         '--out', str(out / 'clusters'), '--eps', str(args.eps), '--min-samples', str(args.min_samples)]
    ]
    for command in commands:
        print(f'\nRunning {command[0]}', flush=True)
        subprocess.run([sys.executable, str(scripts / command[0]), *command[1:]], check=True)
    print(f'\nWorkflow complete. Outputs: {out}')


if __name__ == '__main__':
    main()
