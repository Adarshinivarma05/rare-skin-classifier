import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict
import random
from utils.derm_dataset import DermaDataset
from utils.few_shot_dataset import FewShotDataset

# ✅ Check if file exists
def _verify_npz(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ .npz file not found at {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    required_keys = ['train_images', 'val_images', 'train_labels', 'val_labels']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"❌ Key '{key}' missing in {data_path}")
    return data

# ✅ Standard data loader for base training
def get_loaders(data_path='data/dermamnist/dermamnist.npz', batch_size=16, img_size=224):
    data = _verify_npz(data_path)

    # Auto detect channels to apply normalization
    channels = data['train_images'].shape[-1] if len(data['train_images'].shape) == 4 else 1
    if channels == 1:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        normalize
    ])

    train_dataset = DermaDataset(npz_path=data_path, split='train', transform=transform)
    val_dataset = DermaDataset(npz_path=data_path, split='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset, val_dataset

# ✅ Episodic sampler for few-shot learning
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

# ✅ Few-shot loader using FewShotDataset wrapper
def get_few_shot_loaders(data_path='data/dermamnist/dermamnist.npz', n_way=5, k_shot=5, q=5, episodes=100, img_size=224):
    data = _verify_npz(data_path)

    channels = data['train_images'].shape[-1] if len(data['train_images'].shape) == 4 else 1
    if channels == 1:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        normalize
    ])

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
