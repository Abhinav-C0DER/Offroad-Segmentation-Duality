# 🚜 Offroad Terrain Segmentation (SGDR)

![Status](https://img.shields.io/badge/Status-SOTA_Results-success)
![Backbone](https://img.shields.io/badge/Backbone-DINOv2_Base-blue)

> **Team DUAL PERSONALITY** | *Technical Symposium 2026*

A semantic segmentation model optimized for offroad hazards. By using **Stochastic Gradient Descent with Warm Restarts (SGDR)** and **Weighted Loss**, we achieved a **125% performance increase** in detecting small obstacles like logs and rocks compared to standard benchmarks.

---

### 🏆 Key Results
We shattered the standard DINOv2 baseline on difficult classes:

| Class | Standard IoU | **Our SGDR IoU** | **Growth** |
| :--- | :--- | :--- | :--- |
| **Ground Clutter** | 0.1250 | **0.2822** | **+125.7%** 🚀 |
| **Logs** | 0.1580 | **0.2899** | **+83.4%** |
| **Rocks** | 0.2100 | **0.3830** | **+82.3%** |

<p align="center">
 
</p>

---

### 🧠 Methodology
1.  **Warm Restarts (SGDR):** We implemented a Cosine Annealing schedule with restarts at Epoch 11 ($T_0=10, T_{mult}=2$) to escape local minima.
2.  **Cost-Sensitive Learning:** Applied a **5.0x penalty** to rare classes (Logs, Clutter) to force the model to focus on hazards.

<p align="center">
  
</p>

---

### 🚀 Quick Start

```bash
# 1. Clone & Install
git clone [https://github.com/yourusername/offroad-segmentation.git](https://github.com/yourusername/offroad-segmentation.git)
pip install -r requirements.txt

# 2. Run Inference (Pre-trained Strategy 3)
python inference.py --image "test/hazard_sample.jpg" --model "models/strategy3_best.pth"
