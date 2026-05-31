# 🌿 Plant Disease Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A lightweight Convolutional Neural Network (CNN) for classifying plant leaf diseases into **Healthy**, **Powdery Mildew**, and **Rust** — achieving **96.67% validation accuracy** with only **896K parameters**

---

## 🎯 Overview

Plant diseases cause up to **40% annual crop yield loss** globally. Traditional visual inspection is time-consuming, subjective, and requires expert knowledge. This project addresses these challenges by developing an **automated deep learning system** capable of classifying plant leaf diseases from digital images using a custom CNN architecture.

### Objectives
- 🌱 Develop a robust CNN for multi-class plant disease classification
- ⚡ Achieve high accuracy with minimal computational resources (3.42 MB model)
- 🖥️ Create an intuitive **Streamlit UI** for practical deployment
- 🌍 Demonstrate AI potential in **precision agriculture**

---

## ✨ Features

- ✅ **96.67% Validation Accuracy** with early stopping
- ✅ **Lightweight Model** — only 896,323 parameters (3.42 MB)
- ✅ **Fast Training** — converges in ~5 minutes on GPU
- ✅ **3-Class Classification** — Healthy, Powdery, Rust
- ✅ **Interactive Web App** — upload images and get instant predictions
- ✅ **Real-time Visualization** — confidence scores and disease descriptions
- ✅ **GPU Accelerated** — CUDA with cuDNN support

---

## 📊 Dataset

The dataset consists of leaf images categorized into three classes with a balanced distribution:
(https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
| Dataset Split | Healthy | Powdery | Rust | **Total** |
|:-------------:|:-------:|:-------:|:----:|:---------:|
| **Training**  | 458     | 430     | 434  | **1,322** |
| **Validation**| 20      | 20      | 20   | **60**    |
| **Testing**   | 50      | 50      | 50   | **150**   |
| **Grand Total**| 528    | 500     | 504  | **1,532** |

### Preprocessing Pipeline
- 🔄 **Resizing**: All images standardized to `256 × 256` pixels (RGB)
- 📉 **Rescaling**: Pixel values normalized to `[0, 1]` range (divide by 255)
- 🎨 **Data Augmentation**: Rotation, zoom, flip via `Sequential` pipeline
- 📦 **Batch Processing**: 32 images per batch for efficient training

---

## 🧠 Model Architecture

The proposed CNN follows a sequential architecture with alternating **Convolutional** and **MaxPooling** layers, followed by fully connected **Dense** layers for classification.

### Layer-wise Breakdown

| Layer | Type | Output Shape | Parameters |
|:-----:|:----:|:------------:|:----------:|
| Input | Sequential | `(256, 256, 3)` | 0 |
| Rescaling | Normalization | `(256, 256, 3)` | 0 |
| Conv2D | Conv + ReLU | `(254, 254, 32)` | 896 |
| MaxPooling2D | Downsample | `(127, 127, 32)` | 0 |
| Conv2D_1 | Conv + ReLU | `(125, 125, 64)` | 18,496 |
| MaxPooling2D_1 | Downsample | `(62, 62, 64)` | 0 |
| Conv2D_2 | Conv + ReLU | `(60, 60, 64)` | 36,928 |
| MaxPooling2D_2 | Downsample | `(30, 30, 64)` | 0 |
| Conv2D_3 | Conv + ReLU | `(28, 28, 64)` | 36,928 |
| MaxPooling2D_3 | Downsample | `(14, 14, 64)` | 0 |
| Flatten | Vectorize | `(12544)` | 0 |
| Dense | FC + ReLU | `(64)` | 802,880 |
| **Output** | **Softmax** | **`(3)`** | **195** |

### Model Specifications

Total params:        896,323 (3.42 MB)
Trainable params:    896,323 (3.42 MB)
Non-trainable params:      0
Optimizer params:  1,792,648 (6.84 MB)
Full checkpoint:    10.26 MB


### Key Design Decisions
- 🔹 **4 Convolutional Blocks** — progressive feature extraction (32 → 64 → 64 → 64 filters)
- 🔹 **3×3 Kernels** with ReLU activation and valid padding
- 🔹 **2×2 MaxPooling** halves spatial dimensions after each block
- 🔹 **Flatten** converts 14×14×64 feature maps to 12,544-dim vector
- 🔹 **Dense(64)** learns high-level disease representations
- 🔹 **Softmax Output** — 3 neurons for multi-class probability distribution

---

## 📈 Training Results

### Hyperparameters

| Parameter | Value |
|:----------|:------|
| Epochs | 50 (Early Stopping at 13) |
| Batch Size | 32 |
| Optimizer | Adam (LR = 0.001) |
| Loss Function | Categorical Crossentropy |
| Metrics | Accuracy |
| Early Stopping | Patience = 5 (monitoring `val_accuracy`) |
| Model Checkpoint | Save best model automatically |
| Hardware | CUDA + cuDNN 9.1.0.2 |

### Epoch-wise Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Status |
|:-----:|:---------:|:----------:|:-------:|:--------:|:------:|
| 1 | 51.36% | 0.9224 | 63.33% | 0.7337 | 📈 |
| 2 | 67.55% | 0.6981 | 78.33% | 0.5658 | 📈 |
| 3 | 86.54% | 0.4177 | 91.67% | 0.3404 | 📈 |
| 4 | 90.09% | 0.2660 | 90.00% | 0.2319 | — |
| 5 | 92.89% | 0.2269 | 95.00% | 0.2848 | 📈 |
| 6 | 92.89% | 0.2269 | 90.00% | 0.2706 | — |
| 7 | 92.13% | 0.2341 | 95.00% | 0.1885 | — |
| **8** | **94.10%** | **0.1586** | **96.67%** | **0.1227** | ⭐ **BEST** |
| 9 | 94.10% | 0.1782 | 96.67% | 0.1419 | — |
| 10 | 93.57% | 0.1968 | 96.67% | 0.1097 | — |
| 11 | 94.70% | 0.1600 | 96.67% | 0.1586 | — |
| 12 | 95.84% | 0.1132 | 71.67% | 1.1866 | ⚠️ Overfitting |
| 13 | 94.93% | 0.1518 | 96.67% | 0.0979 | 🛑 Early Stop |

> **Early Stopping** triggered at Epoch 13, restoring weights from **Epoch 8** (best validation accuracy).

### Final Evaluation

```bash
2/2 [==============================] - 1s 558ms/step - accuracy: 0.9667 - loss: 0.1227
Loaded Model Accuracy: 96.67%
