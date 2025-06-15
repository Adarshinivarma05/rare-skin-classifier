from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_dataloaders(batch_size=32, val_split=0.2):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5])
    ])

    full_dataset = DermaMNIST(split='train', transform=transform_train, download=True)
    test_dataset = DermaMNIST(split='test', transform=transform_test, download=True)

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=full_dataset.labels)
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    )






