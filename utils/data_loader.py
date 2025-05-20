from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_dataloaders(batch_size=64, val_split=0.1, seed=42, shuffle=True, image_size=224):
    info = INFO['dermamnist']
    DataClass = DermaMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB
    ])

    # Load datasets
    full_train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    # Train/Validation split
    total_train = len(full_train_dataset)
    val_size = int(val_split * total_train)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# HARSHAA 4.0 🚀
