print("
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, val_split=0.1):
    info = INFO['dermamnist']
    DataClass = DermaMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load full training and test datasets
    full_train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    # Split full train into train and validation
    total_train = len(full_train_dataset)
    val_size = int(val_split * total_train)
    train_size = total_train - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

#HARSHAA 2.0 
      
      ")

