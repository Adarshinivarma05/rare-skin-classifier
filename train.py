from medmnist import DermaMNIST
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader

info = INFO['dermamnist']
DataClass = DermaMNIST

data_transform = transforms.Compose([
transforms.ToTensor(),
transforms.Resize((224, 224)), # for ResNet or ProtoPNet
transforms.Normalize(mean=[.5], std=[.5])
])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
val_dataset = DataClass(split='val', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Your actual code here
