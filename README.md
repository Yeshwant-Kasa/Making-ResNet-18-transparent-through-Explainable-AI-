# Making-ResNet-18-transparent-through-Explainable-AI-
ResNet-18 fine-tuned on CIFAR-10 subset (3000 train/900 test; 128×128) with Captum-based XAI: Saliency, Grad-CAM, Integrated Gradients. Compares custom Parzen Window (Gaussian kernel, h=0.05–1.5) vs. scikit-learn KNN (k=1–20) on Iris (PCA→2D). Visualizes training curves, attribution maps, accuracy vs. params, and decision boundaries.
