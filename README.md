# 🔍 XAI: Deep & Classical Model Interpretability

> **Fine-tunes ResNet-18 on CIFAR-10 using Captum XAI** and compares **custom Parzen Window** vs. **scikit-learn KNN** on PCA-reduced Iris dataset.  
> Includes **full training curves, attribution maps, accuracy sweeps, and decision boundaries** — all coded from scratch in PyTorch and NumPy.

---

## 🧩 Tasks Overview

| Task | Model | Dataset | Key Outputs |
|------|--------|----------|--------------|
| **1** | ResNet-18 (pretrained) | CIFAR-10 subset *(airplane, automobile, bird)* | Training curves, Saliency, Grad-CAM, Integrated Gradients |
| **2** | Parzen (Gaussian Kernel), KNN | Iris → PCA 2D | Accuracy vs. *h/k*, smooth vs. jagged decision boundaries |

---

## ⚙️ Environment & Compatibility

> **Device:** CUDA if available, else CPU  
> **Explicit version pinning** ensures reproducibility (see *Task-A.ipynb*)


## Optimizer: Adam(lr=0.001)

## Loss: CrossEntropy

  Training: 5 epochs

  Epoch| TrainLoss| Train Acc |	Val Acc
  1	   |  0.5100  | 82.07%    |	88.56%
  5	   |  0.1685  | 93.90%    | 90.89%

## Explainability (via [Captum]([url](https://captum.ai/)))
| Method                   | Description                               | Insight                                                           |
| :----------------------- | :---------------------------------------- | :---------------------------------------------------------------- |
| **Saliency**             | Raw input gradients → noisy edge emphasis | Model focuses on object edges                                     |
| **Integrated Gradients** | 20-step baseline integration              | Sharp edges and contours highlighted                              |
| **Grad-CAM**             | Final conv layer activations              | Smooth, class-specific attention on shapes (wings, wheels, beaks) |



## 🔢 Task 2 — Parzen vs. KNN (Task-B.ipynb)
## 📘 Dataset

Iris (3 classes) → StandardScaler → PCA(2D)

Split: 70/30 stratified → 105 train / 45 test

# 🔬 Parzen Window (Custom Implementation)

Gaussian kernel density estimation

Bandwidth sweep: h = 0.05 → 1.5 (15 values)

Peak accuracy: ~93% at h ≈ 0.3–0.5

Boundary: smooth, probabilistic regions

# 🧮 K-Nearest Neighbors

Model: sklearn.neighbors.KNeighborsClassifier

k sweep: 1 → 20

Peak accuracy: ~93% at k = 3–5

Boundary: jagged, locally adaptive

# 📊 Visualization Highlights

# 📈 Accuracy plots: show smooth degradation trends

# 🗺️ Decision boundaries: 300×300 meshgrid, coolwarm colormap

# 🎨 Scatter plots: true labels overlayed for class-wise interpretability
| Category        | Tools / Libraries                  |
| --------------- | ---------------------------------- |
| Deep Learning   | 🧠 PyTorch, Captum                 |
| Classical ML    | 🧮 scikit-learn, NumPy             |
| Data Handling   | pandas, torchvision                |
| Visualization   | Matplotlib, Seaborn                |
| Reproducibility | Version pinning, notebook metadata |
| Environment     | CUDA / CPU adaptive setup          |


```bash
torch==2.2.0
captum==0.7.0
numpy==1.26.4
tensorflow==2.15.0
matplotlib>=3.8
