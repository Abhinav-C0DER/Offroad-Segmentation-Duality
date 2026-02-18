# 🚜 Offroad Terrain Segmentation: Advanced Methodology Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![DINOv2](https://img.shields.io/badge/Backbone-DINOv2-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Team DUAL PERSONALITY** | *B.Tech CSE (AI & ML) - Technical Symposium 2026*

A robust semantic segmentation model designed for unstructured offroad environments. This project leverages **Stochastic Gradient Descent with Warm Restarts (SGDR)** and **Cost-Sensitive Learning** to tackle extreme class imbalance and detect small hazards like logs and rocks with high precision.

---

## 📊 Key Results
Our methodology significantly outperformed standard DINOv2 benchmarks, specifically in minority class detection.

| Class Parameter | Standard Benchmark | **Our Methodology (SGDR)** | **Growth** |
| :--- | :--- | :--- | :--- |
| **Ground Clutter** | 0.1250 | **0.2822** | **+125.7%** 🚀 |
| **Logs** | 0.1580 | **0.2899** | **+83.4%** |
| **Rocks** | 0.2100 | **0.3830** | **+82.3%** |
| **mIoU** | 0.3340 | **0.5346** | **State-of-the-Art** |

<p align="center">
  <img src="path/to/your/nano_style_bar_graph.png" width="800" alt="Performance Comparison Graph">
</p>

---

## 🧠 Methodology: SGDR & Weighted Loss
Standard training often plateaus when facing extreme class imbalance (e.g., Sky vs. Ground Clutter). We implemented a **non-monotonic learning rate schedule** to force the model out of local minima.

### 1. Warm Restarts (SGDR)
We used a Cosine Annealing schedule with restarts at **Epoch 11** ($T_0=10, T_{mult}=2$). This "shake" effect allows the model to re-explore the loss landscape and refine weights for hard-to-detect features.

<p align="center">
  <img src="path/to/your/Code_Generated_Image.png" width="700" alt="SGDR Learning Rate Schedule">
</p>

### 2. Weighted Loss
A **5.0x penalty multiplier** was applied to the Cross-Entropy Loss for rare classes:
* `Logs`
* `Ground Clutter`
* `Rocks`

---

## 🛠️ Architecture
* **Backbone:** [DINOv2-Base (ViT-B/14)](https://github.com/facebookresearch/dinov2) - Frozen, self-supervised weights.
* **Decoder:** Custom **ConvNeXt-based head** for localized feature extraction.
* **Resolution:** $518 \times 518$ input ($37 \times 37$ spatial tokens).
* **TTA:** Test Time Augmentation (Horizontal Flip) enabled for final inference.

---

## 📸 Qualitative Analysis
The impact of SGDR is visible in the sharpness of hazard boundaries.

| Input Image | Ground Truth | **SGDR Prediction** |
| :---: | :---: | :---: |
| <img src="path/to/input_sample.png" width="200"> | <img src="path/to/gt_sample.png" width="200"> | <img src="path/to/pred_sample.png" width="200"> |

> *Notice the precise segmentation of fallen logs (brown) and ground clutter (olive) in the SGDR Prediction, which are typically blurred by standard models.*

---

## 🚀 Installation & Usage

### Prerequisites
* NVIDIA GPU (T4 or better recommended)
* Python 3.8+

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/offroad-segmentation-sgdr.git](https://github.com/yourusername/offroad-segmentation-sgdr.git)
cd offroad-segmentation-sgdr
