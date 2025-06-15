import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def compute_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return vis



