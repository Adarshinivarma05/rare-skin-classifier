import torch
import numpy as np
import cv2

def generate_gradcam(model, input_tensor, target_class, feature_module, device):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = feature_module.register_forward_hook(forward_hook)
    handle_b = feature_module.register_backward_hook(backward_hook)

    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    output = model(input_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    grad = gradients[0].squeeze()
    act = activations[0].squeeze()

    weights = torch.mean(grad, dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (224, 224))

    handle_f.remove()
    handle_b.remove()

    return cam
