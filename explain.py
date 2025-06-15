import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.explain_utils import compute_gradcam
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=1)
imgs, lbls = next(iter(test_loader))
img = imgs[0].to(device)
vis = compute_gradcam(model, img.unsqueeze(0), model.backbone.layer4[-1])
plt.imshow(vis)
plt.axis('off')
plt.savefig('gradcam_example.png')





