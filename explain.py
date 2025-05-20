import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from medmnist import INFO, DermaMNIST
from models.protopnet_skin_classifier import ProtoPNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.explain_utils import get_top_k_prototypes, overlay_activation_map, show_prototype_patch
from utils.visualizer import visualize_prototypes

# Dataset setup
info = INFO['dermamnist']
DataClass = DermaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_dataset = DataClass(split='test', transform=data_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Model & device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet(num_prototypes=30, num_classes=7).to(device)
model.load_state_dict(torch.load('protopnet_skin.pt'))
model.eval()

# Grad-CAM setup
target_layers = [model.backbone.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type == 'cuda'))

for images, labels in test_loader:
    images = images.to(device)

    # 1. Grad-CAM
    grayscale_cam = cam(input_tensor=images)[0]
    img = images.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    cam_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title(f"Input Image (Label: {labels.item()})")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(cam_img)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')

    # 2. Prototype Activations and Visualization
    features, logits, prototype_activations = model.push_forward(images)
    topk_indices, topk_values = get_top_k_prototypes(prototype_activations[0], k=3)

    plt.subplot(1,3,3)
    # Show top prototype patch (only first top prototype)
    top_prototype = model.prototype_vectors[topk_indices[0]]
    top_prototype = top_prototype.view(3, model.prototype_shape[2], model.prototype_shape[3])
    show_prototype_patch(top_prototype, title=f"Top Prototype #{topk_indices[0]}")

    plt.tight_layout()
    plt.show()

    break  # Only visualize one example

# Optional: visualize multiple prototypes on test set
# visualize_prototypes(model, test_loader, device, num_images=3)
