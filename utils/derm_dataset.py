import numpy as np
import torch
from torch.utils.data import Dataset

class DermaDataset(Dataset):
    def __init__(self, npz_path, split='train', transform=None):
        data = np.load(npz_path)

        if split == 'train':
            self.images = data['train_images']
            self.labels = data['train_labels']
        elif split == 'val':
            self.images = data['val_images']
            self.labels = data['val_labels']
        else:
            self.images = data['test_images']
            self.labels = data['test_labels']

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].item()

        if self.transform:
            image = self.transform(image)

        return image, label
