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

```bash
torch==2.2.0
captum==0.7.0
numpy==1.26.4
tensorflow==2.15.0
matplotlib>=3.8
