import torch
import matplotlib.pyplot as plt

def imshow(img_tensor, mean=0.5, std=0.5, title=None):
    """
    Unnormalize and show an image tensor [C, H, W].
    """
    img = img_tensor.cpu().clone()
    img = img * std + mean
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_prototypes(model, data_loader, device, num_images=5):
    """
    For a few images, show the input and its top prototype.
    Requires model to implement a 'push_forward' that returns
    prototype activations.
    """
    model.eval()
    shown = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            features, logits, prototype_activations = model.push_forward(imgs)
            for i in range(imgs.size(0)):
                if shown >= num_images:
                    return
                top_idx = torch.argmax(prototype_activations[i]).item()
                prototype = model.prototype_vectors[top_idx]
                plt.figure(figsize=(6, 3))

                plt.subplot(1, 2, 1)
                imshow(imgs[i], title=f"Input (Label: {labels[i].item()})")

                plt.subplot(1, 2, 2)
                imshow(prototype, title=f"Prototype #{top_idx}")

                plt.tight_layout()
                plt.show()
                shown += 1
