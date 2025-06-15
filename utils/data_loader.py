# utils/data_loader.py
from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


def load_dermamnist_dataset(batch_size=32, val_split=0.2):
    """
    Returns train/val/test dataloaders, num_classes, and flattened train labels
    for class‑weight computation.
    """
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    full_train_ds = DermaMNIST(split="train", transform=tf_train, download=True)
    test_ds       = DermaMNIST(split="test",  transform=tf_eval,  download=True)

    # Stratified split for validation
    idx = list(range(len(full_train_ds)))
    tr_idx, val_idx = train_test_split(
        idx, test_size=val_split,
        stratify=full_train_ds.labels, random_state=42
    )
    train_ds = Subset(full_train_ds, tr_idx)
    val_ds   = Subset(full_train_ds, val_idx)
    val_ds.dataset.transform = tf_eval  # eval transforms for val

    kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    num_classes = 7
    flat_labels = np.array(full_train_ds.labels).squeeze()

    return train_loader, val_loader, test_loader, num_classes, flat_labels

