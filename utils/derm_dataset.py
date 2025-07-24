import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DermaDataset(Dataset):
    def __init__(self, root='data/dermamnist', split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root, f'{split}.csv'))  # Expects train.csv, val.csv, etc.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root, row['image'])  # column must be named 'image'
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])  # column must be named 'label'

        if self.transform:
            image = self.transform(image)

        return image, label
