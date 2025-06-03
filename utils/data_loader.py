from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def get_dataloaders(batch_size=64):
   info = INFO['dermamnist']
   DataClass = DermaMNIST

   train_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(20),
       transforms.ToTensor(),
       transforms.Normalize(mean=[.5], std=[.5])
   ])

   test_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[.5], std=[.5])
   ])

   train_dataset = DataClass(split='train', transform=train_transform, download=True)
   test_dataset = DataClass(split='test', transform=test_transform, download=True)

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   return train_loader, test_loader

