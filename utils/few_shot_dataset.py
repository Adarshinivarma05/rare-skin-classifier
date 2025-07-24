# utils/few_shot_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import random

class FewShotDataset(Dataset):
    def __init__(self, data, labels, n_way=5, k_shot=5, q_queries=15, episodes=1000, split='train'):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.episodes = episodes
        self.split = split
        self.label_set = list(set(labels.tolist()))
        self.class_to_indices = {
            label: np.where(labels == label)[0].tolist()
            for label in self.label_set
        }

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        selected_classes = random.sample(self.label_set, self.n_way)

        for i, class_label in enumerate(selected_classes):
            indices = self.class_to_indices[class_label]
            selected_indices = random.sample(indices, self.k_shot + self.q_queries)
            support_idx = selected_indices[:self.k_shot]
            query_idx = selected_indices[self.k_shot:]

            support_images.extend(self.data[support_idx])
            support_labels.extend([i] * self.k_shot)

            query_images.extend(self.data[query_idx])
            query_labels.extend([i] * self.q_queries)

        support_images = torch.tensor(np.stack(support_images), dtype=torch.float32)
        support_labels = torch.tensor(support_labels)
        query_images = torch.tensor(np.stack(query_images), dtype=torch.float32)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels
