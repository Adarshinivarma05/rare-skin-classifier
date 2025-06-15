import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


def compute_gradcam(model, input_tensor, target_layer, use_cuda: bool = True):
    """
    Returns an RGB numpy image with the Grad‑CAM overlay.
    """
    model.eval()
    target_class = torch.argmax(model(input_tensor)).item()
    targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # [H, W]

    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1]

    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return vis

