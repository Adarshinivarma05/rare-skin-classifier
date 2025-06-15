from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def _mixup(batch_x, batch_y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx  = torch.randperm(batch_x.size(0))
    mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
    y_a, y_b = batch_y, batch_y[idx]
    return mixed_x, y_a, y_b, lam

def get_dataloaders(batch_size=32, val_split=0.2):
    tr_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
        transforms.ToTensor(), transforms.Normalize([.5],[.5])
    ])
    ev_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), transforms.Normalize([.5],[.5])
    ])

    full = DermaMNIST(split='train', transform=tr_tf, download=True)
    test = DermaMNIST(split='test',  transform=ev_tf, download=True)

    idx = list(range(len(full)))
    tr_i, va_i = train_test_split(idx, test_size=val_split,
                                  stratify=full.labels, random_state=42)
    train_set, val_set = Subset(full, tr_i), Subset(full, va_i)
    val_set.dataset.transform = ev_tf            # eval transforms

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(train_set, shuffle=True,  **kw),
            DataLoader(val_set,   shuffle=False, **kw),
            DataLoader(test,      shuffle=False, **kw),
            _mixup)   # return mixup fn



