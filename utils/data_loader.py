from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch


def get_loaders(batch_size: int = 32, val_split: float = 0.2):
    """Return train/val/test DataLoaders + flattened train labels."""
    aug_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])

    aug_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    full = DermaMNIST(split="train", transform=aug_train, download=True)
    test = DermaMNIST(split="test",  transform=aug_eval,  download=True)

    idx = np.arange(len(full))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_split, stratify=full.labels, random_state=42
    )
    train_ds = Subset(full, tr_idx)
    val_ds   = Subset(full, va_idx)
    val_ds.dataset.transform = aug_eval   # eval transforms

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test,     shuffle=False, **kw),
        np.array(full.labels).squeeze()        # flat labels for class weights
    )
