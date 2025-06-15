import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from medmnist import DermaMNIST

labels = np.array(DermaMNIST(split='train', download=True).labels).squeeze()

# ✅ new API‑safe call (works for all scikit‑learn versions)
class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels)

criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device),
        label_smoothing=0.1)



