import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np


def gradcam_vis(model, img_tensor, target_layer, device):
    model.eval()
    target_class = torch.argmax(model(img_tensor.to(device))).item()
    cam = GradCAM(model, [target_layer], use_cuda=device.type == "cuda")
    grayscale_cam = cam(img_tensor.to(device), [ClassifierOutputTarget(target_class)])[0]
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return show_cam_on_image(img, grayscale_cam, use_rgb=True)
