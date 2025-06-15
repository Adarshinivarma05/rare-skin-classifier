import torch, matplotlib.pyplot as plt
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.explain_utils import gradcam_vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProtoPNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))

_, _, test_loader, _ = get_loaders(batch_size=1)
img, _ = next(iter(test_loader))
vis = gradcam_vis(model, img.to(device), model.backbone.layer4[-1], device)
plt.imshow(vis); plt.axis("off"); plt.savefig("gradcam_example.png")
print("Grad‑CAM saved → gradcam_example.png")
