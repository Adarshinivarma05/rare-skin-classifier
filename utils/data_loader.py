import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import random
from collections import defaultdict
from PIL import Image

class FewShotDataset(Dataset):
    def __init__(self, data, n_way, k_shot, q, transform=None):
        self.data = data
        self.n_way = n_way
        self.k_shot = k_shot
        self.q = q
        self.transform = transform
        self.classes = list(data.keys())

    def __len__(self):
        return 100  # number of episodes per epoch

    def __getitem__(self, idx):
        sampled_classes = random.sample(self.classes, self.n_way)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        label_map = {cls: i for i, cls in enumerate(sampled_classes)}

        for cls in sampled_classes:
            imgs = self.data[cls]
            selected = random.sample(imgs, self.k_shot + self.q)
            support_imgs = selected[:self.k_shot]
            query_imgs = selected[self.k_shot:]

            for img_path in support_imgs:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(label_map[cls])

            for img_path in query_imgs:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(label_map[cls])

        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels),
        )

def build_class_image_dict(data_dir):
    class_image_dict = defaultdict(list)
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            class_image_dict[cls].append(img_path)
    return class_image_dict

def get_few_shot_loaders(data_dir="data/dermamnist", n_way=5, k_shot=1, q=5, episodes=100, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    class_image_dict = build_class_image_dict(data_dir)
    dataset = FewShotDataset(class_image_dict, n_way, k_shot, q, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from collections import defaultdict
import random
from torch.utils.data import Sampler

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

from torchvision import transforms
from .derm_dataset import DermaDataset  # adjust if needed
from torch.utils.data import DataLoader

def get_few_shot_loaders(n_way, k_shot, q, episodes, root='data/dermamnist', img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = DermaDataset(root=root, transform=transform)
    labels = [label for _, label in dataset]

    sampler = EpisodicBatchSampler(labels, n_way, k_shot, q, episodes)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    return loader, loader  # using same loader for train/val for simplicity





import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
import random
from utils.derm_dataset import DermaDataset

# ✅ Standard dataloader
def get_loaders(npz_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = DermaDataset(npz_path=npz_path, split='train', transform=transform)
    val_dataset = DermaDataset(npz_path=npz_path, split='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ✅ Few-shot episodic batch sampler
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

# ✅ Few-shot dataloader for episodic training
def get_few_shot_loaders(npz_path, n_way, k_shot, q_queries, episodes):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = np.load(npz_path)
    train_images = data['train_images']
    train_labels = data['train_labels'].flatten()

    # build class-wise mapping
    class_to_indices = {i: [] for i in range(n_way)}
    for idx, label in enumerate(train_labels):
        if label in class_to_indices:
            class_to_indices[label].append(idx)

    episodes_data = []
    for _ in range(episodes):
        support_set = []
        query_set = []
        selected_classes = np.random.choice(list(class_to_indices.keys()), n_way, replace=False)
        for cls in selected_classes:
            indices = np.random.choice(class_to_indices[cls], k_shot + q_queries, replace=False)
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]
            for idx in support_idx:
                support_set.append((train_images[idx], cls))
            for idx in query_idx:
                query_set.append((train_images[idx], cls))
        episodes_data.append((support_set, query_set))

    return episodes_data



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

