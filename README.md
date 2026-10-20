# **SimCLR-for-Ionospheric-Event-Identification**
### **Using Self-Supervised Machine Learning to Identify Pulsed Ionospheric Flows in SuperDARN Observations**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker Ready](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
---

## **Pipeline Overview**
<p align="center">
  <img width="440" height="617" alt="Screenshot 2025-11-23 192609" src="https://github.com/user-attachments/assets/46e4d31b-eb14-4bfd-a66b-4e1caa3c30f5" />
</p>

This project implements a pipeline for identifying Pulsed Ionospheric Flows (PIFs) from SuperDARN radar data using self-supervised contrastive learning (SimCLR). The pipeline applies standard SimCLR with an additional attention layer in the encoder. Augmented views of an observed sample are created, encoded by a CNN and then clustered into a learned embedding space to identify PIF behaviour. 

---
## Repository structure

| Path | Purpose |
|---|---|
| `src/ionospheric_simclr/models.py` | Defines the CNN encoder, multi-head attention layer, projection head, and complete SimCLR model.|
| `src/ionospheric_simclr/data_loader.py` | Loads HDF5 segments, applies augmentations, normalises power values and forms SimCLR batches. |
| `src/ionospheric_simclr/augmentation_strategies.py` | Implements all stochastic transformations used to create positive pairs.|
| `src/ionospheric_simclr/train.py` | Full training loop with logging, early stopping and checkpointing. |
| `scripts/` | Command-line entry points for each stage |
| `docs/` | Reports|
| `tests/` | Test stubs |
| `Dockerfile` ||


## **Project Motivation**

The aim of this pipeline is to develop an automated method for detecting Pulsed Ionospheric Flows (PIFs), which are short poleward moving flows in the ionosphere that are signatures of magnetic reconnection in the magnetosphere and transfer plasma into Earth's magnetosphere. These signatures can be detected in the ionosphere using high frequency coherent scatter radars such as SuperDARN. Previous studies have aimed to analyse and classify observed PIFs to help answer questions about magnetospheric convection. However, manual classifciation provbes time-consuming and does not provide the volume of classiications needed to answer high level questions regarding magnetospheric dynamics. Supervised classification also remains a challenge due to the large amount of unlabelled data.

This project aims to build upon previous studies by applying SimCLR to learn spatio-temporal structure directly from unlabelled radar data, allowing PIF features to emerge in the learned embedding space.

---

## **Project Background**

### **Reconnection and the Dungey Cycle**

Magnetic reconnection occurs when solar wind plasma penetrates Earths magnetosphere. As the plasma enters the magnetosphere, a Flux Transfer Event (FTE) occurs, which drives a convection process in the magnetosphere known as the Dungey Cycle. During this convection process magnetic field lines are dragged down the magnetotail. As they have footprints on the Earth, the motion of the field lines drive Poleward Moving Auroral Forms (PMAFs) in the ionosphere. 

<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/5e14a02d-57ad-4356-a958-c1db3be7f9d2" />
</p>

---

### **SuperDARN**

These events can be detected by High Frequency coherent scatter radars such as SuperDARN (The Super Dual Auroral Radar Network). SuperDARN covers extensive regions of the Northern and Southern hemisphere. Each radar consits of 16 different antennas (look directions). SuperDARN detects radio wave reflections from magnetic field aligned plasma irregularities in the ionosphere. This allows for PMAFs to be observed in the ionosphere as shown below. 

<p align="center">
  <img width="650" src="https://github.com/user-attachments/assets/e507c60c-3c1b-42e7-b4d8-f303693fd4eb" />
</p>

---

## **Pre-Processing**

This project uses data from the SuperDARN Hankasalmi radar in Finland between 19/06/1995 00:00 - 1996/01/00 00:00 consisting of 2245 2 hour observations. Data is stored in compressed .bz2 files, and loaded using pyDARN. Records are extracted and filtered by beam numbers, with each observation containing the time, range gates, backscattered power, Doppler velocity and spectral width. Missing values are filled with negative padding values and each radar beam is processes separately. 

Range gates differ per beam, therefore each gate is mapped into AACGM magnetic latitude, producing a consistent spatial axis across all beams. As PIFs occur on approximately 10 minute scales, data is segmented into 1-hour windows, and resampled to 20 time steps, M magnetic-latitude beams and 3 channels including power, doppler velocity and spctral width. 

Segments are stored in an HDF5 file and split into train/validation/test sets.

---

# **SimCLR Framework**

SimCLR is used to learn representations of PIF samples by contrasting augmented views of each segment. Each segment x is transformed twice using random augmentations to create two correlated views x_i and x_j. The augmentations used are:

- Random Noise  
- Scaling  
- Saturation  
- Masking  
- Swapping  
- Translation  

<p align="center">
  <img width="500" height="800" alt="augmentations (1)" src="https://github.com/user-attachments/assets/12baaeaa-0c6e-4499-b004-fe9e84401171" />
</p>


## **Base Encoder**

The augmented samples are passed through a CNN responsible for extracting representation vectors from the agumented samples. An optional multi-head attention layer is included, which is used to model long range relationships while masking invalid padded regions.

---

## **Projection Head**

A Multi Layer Perceptron (MLP) is chosen to project the representation vectors into a lower dimensional space suitable for the NT-Xent contrasrive loss function.

---

## **Loss Function**

NT-Xent loss is used to carry out contrastive prediction. The loss maximises the similairyt between 2 augmented views of the same segment and reduces similarity with all other samples in the batch. This shapes the representation space such that physically similar patterns lie close together. 

---

# **Accuracy Metrics & Clustering**

## **Top-K Accuracy**

The percentage of times the correct pair x_j is identified in the top K embeddings nearest to x_i. Top-k accuracy is computed at the end of each epoch using the valildation set. 

---

## **DBSCAN Clustering**

DBSCAN clustering is used to detect high density regions in the embedding space.  
A moderately coherent cluster (silhouette score = 0.53) emerges containing:

- 30% PIF-like events  
- 52% Groundscatter  
- 18% Non-Events  

This cluster contains short, intense poleward-moving structures consistent with PIF characteristics. Future work should aim to implement multi-modal features.

---



**Workflow:** download processed data, inspect observations, train SimCLR,
extract encoder embeddings, cluster and visualise.

[Dataset](https://www.kaggle.com/datasets/amanbhalla01/superdarn-hankasalmi-processed-segments)
· [Project report](docs/report.pdf)


## Installation

Use Python **3.10+**

### 1. Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CPU-only PyTorch on Linux or Windows:

```bash
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the installation command selected for your system at
[PyTorch installation](https://pytorch.org/get-started/locally/).

Install the other dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Dataset Download

**[SuperDARN Hankasalmi — Processed Segments](https://www.kaggle.com/datasets/amanbhalla01/superdarn-hankasalmi-processed-segments)**

The workflow expects `train.h5`, `val.h5` and `test.h5`. The current model uses the **power
channel**, the preprocessing code also stores Doppler velocity and spectral
width. Missing samples use `-9999`.

Install the Kaggle CLI and login:

```bash
python -m pip install --upgrade kaggle
kaggle auth login
```

Download and unzip:

```bash
kaggle datasets download -d amanbhalla01/superdarn-hankasalmi-processed-segments -p data --unzip
```


## Run the workflow

Inspect and train on up to 64 segments per split, then embed and cluster up to
64 test segments:

```bash
python scripts/run_workflow.py --data-dir data --out-dir runs/quickstart --epochs 1 --batch-size 8 --max-samples 64 --min-samples 5
```


### 1. Inspect processed observations

```bash
python scripts/inspect_data.py --input data/train.h5 --out runs/inspection
```

Saves `samples.png` and `summary.json`.

### 2. Train SimCLR

```bash
python scripts/train_ssl.py --train data/train.h5 --val data/val.h5 --out runs/ssl --epochs 20 --batch-size 16
```

The model uses CUDA when available and otherwise CPU. Select explicitly with
`--device cpu` or `--device cuda`.

Outputs include:

- `runs/ssl/checkpoints/best_model.pth`
- `runs/ssl/config.json`
- `runs/ssl/metrics.csv`
- `runs/ssl/loss.png`
- `runs/ssl/logs/training.log`


### 3. Extract encoder embeddings

```bash
python scripts/extract_embeddings.py --input data/test.h5 --checkpoint runs/ssl/checkpoints/best_model.pth --out runs/embeddings/test.npz
```


### 4. Cluster and visualise

```bash
python scripts/cluster_dbscan.py --embeddings runs/embeddings/test.npz --out runs/clusters --eps 0.4 --min-samples 20
```

DBSCAN runs in the full embedding space. PCA is used only to display the result.
Adjust `--eps` and `--min-samples` for the number and distribution of observations.

Outputs:

- `pca_clusters.png`: projected embeddings coloured by cluster.
- `assignments.csv`: segment IDs, cluster assignments, PCA coordinates and metadata.
- `cluster_examples.png`: radar samples from up to four of the largest clusters,
- `summary.json`: cluster count, noise count and a silhouette score


## Docker

Install [Docker](https://docs.docker.com/get-started/get-docker/) and download the
dataset.

### Build

```bash
docker build -t ionospheric-simclr .
```


### Run — Windows PowerShell

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\data,target=/data,readonly" `
  --mount "type=bind,source=$($PWD.Path)\runs,target=/outputs" `
  ionospheric-simclr `
  python scripts/run_workflow.py --data-dir /data --out-dir /outputs/docker-run --device cpu --epochs 1 --batch-size 8 --max-samples 64 --min-samples 5
```

Results appear in `runs/docker-run/`



