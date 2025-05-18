import torch
from torchvision import transforms
from medmnist import INFO, DermaMNIST
from torch.utils.data import DataLoader
from models.protopnet_skin_classifier import ProtoPNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np

info = INFO['dermamnist']
DataClass = DermaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_dataset = DataClass(split='test', transform=data_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet(num_prototypes=30, num_classes=7).to(device)
model.load_state_dict(torch.load('protopnet_skin.pt'))
model.eval()

target_layers = [model.backbone.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type=='cuda'))

for images, labels in test_loader:
    images = images.to(device)
    grayscale_cam = cam(input_tensor=images)[0, :]
    img = images.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.title(f"Label: {labels.item()}")
    plt.axis('off')
    plt.show()
    break  # Show one example only

# Your actual code here
