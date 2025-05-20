import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def imshow(img_tensor, mean=0.5, std=0.5, title=None):
    """Utility to unnormalize and display an image tensor"""
    img = img_tensor.cpu().clone()
    img = img * std + mean  # Unnormalize
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_prototypes(model, data_loader, device, num_images=5):
    """
    Shows a few input images and the closest prototype that was activated.
    """
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features, logits, prototype_activations = model.push_forward(images)
            # shape: (batch_size, num_prototypes)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    return

                plt.figure(figsize=(6, 3))

                # Show input image
                plt.subplot(1, 2, 1)
                imshow(images[i])
                plt.title(f"Input (Label: {labels[i].item()})")

                # Show top-activated prototype
                top_prototype_idx = torch.argmax(prototype_activations[i]).item()
                prototype = model.prototype_vectors[top_prototype_idx]
                prototype_img = prototype.view(3, model.prototype_shape[2], model.prototype_shape[3])
                plt.subplot(1, 2, 2)
                imshow(prototype_img)
                plt.title(f"Prototype {top_prototype_idx}")

                plt.tight_layout()
                plt.show()
                images_shown += 1
