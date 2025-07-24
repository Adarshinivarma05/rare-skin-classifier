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
