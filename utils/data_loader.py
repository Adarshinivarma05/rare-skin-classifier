import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict
import random
from utils.derm_dataset import DermaDataset
from utils.few_shot_dataset import FewShotDataset

# ✅ Standard data loader for normal training
def get_loaders(data_path='data/dermamnist/dermamnist.npz', batch_size=16, img_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5], [0.5]) if np.load(data_path)['train_images'].shape[-1] == 1 else
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = DermaDataset(npz_path=data_path, split='train', transform=transform)
    val_dataset = DermaDataset(npz_path=data_path, split='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset, val_dataset

# ✅ Episodic batch sampler for few-shot
class EpisodicBatchSampler(Sampler):
    def __init__(self, labels, n_way, k_shot, q, episodes):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q = q
        self.episodes = episodes
        self.class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            batch = []
            selected_classes = random.sample(self.classes, self.n_way)
            for cls in selected_classes:
                indices = random.sample(self.class_to_indices[cls], self.k_shot + self.q)
                batch.extend(indices)
            yield batch

# ✅ Few-shot episodic loaders
def get_few_shot_loaders(data_path='data/dermamnist/dermamnist.npz', n_way=5, k_shot=5, q=5, episodes=100, img_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5], [0.5]) if np.load(data_path)['train_images'].shape[-1] == 1 else
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load .npz
    data = np.load(data_path)
    train_data = {
        'images': data['train_images'],
        'labels': data['train_labels']
    }
    val_data = {
        'images': data['val_images'],
        'labels': data['val_labels']
    }

    train_dataset = FewShotDataset(train_data, n_way=n_way, k_shot=k_shot, q=q, episodes=episodes, transform=transform)
    val_dataset = FewShotDataset(val_data, n_way=n_way, k_shot=k_shot, q=q, episodes=episodes, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader
