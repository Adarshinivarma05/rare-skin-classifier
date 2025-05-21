from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO
from medmnist.dataset import DermaMNIST

def get_dataloaders(batch_size=64, val_split=0.1):
    data_flag = 'dermamnist'
    info = INFO[data_flag]
    num_classes = len(info['label'])

    # ✅ Normalization based on dataset stats (mean/std from medmnist docs)
    norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])  # or use dataset-specific if known

    # ✅ Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        norm_transform,
    ])

    # ✅ Minimal augmentation for validation/test
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        norm_transform,
    ])

    # ✅ Load datasets
    train_dataset = DermaMNIST(split='train', transform=train_transform, download=True)
    test_dataset = DermaMNIST(split='test', transform=test_transform, download=True)

    # ✅ Manual validation split from training data
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # ✅ DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
