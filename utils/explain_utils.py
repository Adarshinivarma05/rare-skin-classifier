import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms import ToPILImage

def get_top_k_prototypes(activations, k=3):
    """
    Given prototype activations (tensor shape: [num_prototypes]),
    returns indices and values of top-k prototypes.
    """
    topk = torch.topk(activations, k=k)
    return topk.indices.cpu().numpy(), topk.values.cpu().numpy()

def show_prototype_patch(prototype_vector, title="Prototype"):
    """
    Display a single prototype patch (Tensor shape: [C, H, W]).
    """
    prototype_img = prototype_vector.detach().cpu()
    prototype_img = prototype_img * 0.5 + 0.5  # unnormalize
    prototype_img = ToPILImage()(prototype_img)
    plt.imshow(prototype_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def overlay_activation_map(image_tensor, activation_map):
    """
    Overlay activation heatmap onto an image.
    
    image_tensor: Tensor [C, H, W] normalized between 0 and 1
    activation_map: Tensor [H', W'] with activation intensities
    """
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    activation_map = activation_map.detach().cpu().numpy()
    
    # Resize activation map to image size
    activation_map_resized = cv2.resize(activation_map, (image.shape[1], image.shape[0]))
    activation_map_resized = (activation_map_resized - activation_map_resized.min()) / (activation_map_resized.max() - activation_map_resized.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_resized), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0  # BGR to RGB and normalize

    overlayed = 0.5 * image + 0.5 * heatmap
    plt.imshow(overlayed)
    plt.title("Activation Overlay")
    plt.axis('off')
    plt.show()
