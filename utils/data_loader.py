from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_dataloaders(batch_size: int = 128, val_split: float = 0.2):
    """Return train / val / test dataloaders with heavy augmentation."""
    # ---------- Transforms ----------
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    # ---------- Datasets ----------
    full_dataset = DermaMNIST(split='train', transform=transform_train, download=True)
    test_dataset = DermaMNIST(split='test', transform=transform_eval, download=True)

    # ---------- Stratified split ----------
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=full_dataset.labels, random_state=42
    )
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    val_set.dataset.transform = transform_eval      # eval transforms for val

    # ---------- Dataloaders ----------
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

