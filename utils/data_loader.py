# utils/data_loader.py
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def get_dataloaders(batch_size=64, val_split=0.2):
   info = INFO['dermamnist']
   DataClass = DermaMNIST

   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[.5], std=[.5])
   ])

   full_dataset = DataClass(split='train', transform=transform, download=True)
   test_dataset = DataClass(split='test', transform=transform, download=True)

   indices = list(range(len(full_dataset)))
   train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=full_dataset.labels)

   train_set = Subset(full_dataset, train_idx)
   val_set = Subset(full_dataset, val_idx)

   train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   return train_loader, val_loader, test_loader

