# **SimCLR-for-Ionospheric-Event-Identification**
### **Using Self-Supervised Machine Learning to Identify Pulsed Ionospheric Flows in SuperDARN Observations**

---

## **Pipeline Overview**
<p align="center">
  <img width="650" src="https://github.com/user-attachments/assets/5cbdee70-7d4d-4d2d-998a-7cdd83834c44" />
</p>

This project implements a pipeline for identifying Pulsed Ionospheric Flows (PIFs) from SuperDARN radar data using self-supervised contrastive learning (SimCLR). The pipeline applies standard SimCLR with an additional attention layer in the encoder. Augmented views of an observed sample are created, encoded by a CNN and then clustered into a learned embedding space to identify PIF behaviour. 

---

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

---

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
A moderately coherent cluster emerges containing:

- 30% PIF-like events  
- 52% Groundscatter  
- 18% Non-Events  

This cluster contains short, intense poleward-moving structures consistent with PIF characteristics.

---

# **Repository Scripts**

### **pre-processing.py**  
Converts raw fitacf files into magnetic latitude, time-series segments, resamples into 1 hour windows and writes the HDF5 dataset.

### **data_loader.py**  
Loads HDF5 segments, applies augmentations, normalises power values and forms SimCLR batches.

### **augmentation_strategies.py**  
Implements all stochastic transformations used to create positive pairs.

### **models.py**  
Defines the CNN encoder, multi-head attention layer, projection head, and complete SimCLR model.

### **train.py**  
Full training loop with logging, early stopping and checkpointing.

### **visualisation.py**  
Extracts embeddings and plots nearest neighbours of selected samples for qualitative inspection.

### **pca.py**  
Runs PCA, DBSCAN clustering and nearest-neighbour search in the embedding space.
