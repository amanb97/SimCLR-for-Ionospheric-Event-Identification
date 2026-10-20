"""Runtime helpers shared by the command-line workflow."""
import json
import random
from pathlib import Path
import numpy as np


def choose_device(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Select --device cpu.')
    return torch.device(name)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')


def plot_metrics(csv_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=',', names=True))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(data['epoch'], data['train_loss'], marker='o', label='Training')
    ax.plot(data['epoch'], data['val_loss'], marker='o', label='Validation')
    ax.set(xlabel='Epoch', ylabel='NT-Xent loss', title='Contrastive training')
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)
