import torch, matplotlib.pyplot as plt
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.explain_utils import compute_gradcam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=1)
img, _ = next(iter(test_loader))
img = img.to(device)

vis = compute_gradcam(model, img, model.backbone.layer4[-1], use_cuda=device.type=='cuda')
plt.imshow(vis)
plt.axis('off')
plt.savefig('gradcam_example.png')
print("Grad‑CAM saved to gradcam_example.png")




