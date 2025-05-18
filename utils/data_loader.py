from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    info = INFO['dermamnist']
    DataClass = DermaMNIST

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
