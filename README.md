# Rare Skin Condition Classifier (ProtoPNet + Grad‑CAM)

**Goal:** ≥ 90 % test accuracy & weighted F1 ≈ 0.90 on DermaMNIST.

### Highlights
* ResNet‑50 backbone + 90 prototypes
* Heavy AutoAugment / jitter / rotation
* Class‑balanced weighted loss + label smoothing
* AdamW + CosineWarmRestarts + AMP mixed precision
* Early stopping & metric plots
* Grad‑CAM visualization (`explain.py`)

### Usage
```bash
pip install torch torchvision medmnist scikit-learn pytorch-grad-cam
python train.py      # trains & saves best_model.pth
python test.py       # evaluates on test split
python explain.py    # produces gradcam_example.png






