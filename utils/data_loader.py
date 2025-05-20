
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_dataloaders(batch_size=64, val_split=0.1, seed=42, shuffle=True, image_size=224):
    info = INFO['dermamnist']
    DataClass = DermaMNIST

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_train_dataset = DataClass(split='train', transform=transform_train, download=True)
    test_dataset = DataClass(split='test', transform=transform_val, download=True)

    total_train = len(full_train_dataset)
    val_size = int(val_split * total_train)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # Override val transform for validation subset
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
