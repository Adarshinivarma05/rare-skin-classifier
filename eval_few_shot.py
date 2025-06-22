"""
Evaluate a saved ProtoNet on unseen few-shot episodes.
"""

import argparse, numpy as np, torch
from torchvision import transforms, models
from medmnist import DermaMNIST
from utils.episode_loader import get_episode_loader

# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",     type=str,   default="proto_resnet50.pth")
parser.add_argument("--n_way",    type=int,   default=5)
parser.add_argument("--k_shot",   type=int,   default=5)
parser.add_argument("--q",        type=int,   default=1)
parser.add_argument("--episodes", type=int,   default=600)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# dataset (test split)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5])
])
test_ds = DermaMNIST(split="test", transform=transform, download=True)
test_loader = get_episode_loader(test_ds, args.n_way, args.k_shot, args.q,
                                 args.episodes, shuffle=False)

# ---------------------------------------------------------------------
# recreate ProtoNet (encoder only)
class ProtoNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = torch.nn.Identity()
        self.encoder = backbone
    def forward(self, s_img, s_lbl, q_img):
        z_sup = self.encoder(s_img)
        z_qry = self.encoder(q_img)
        classes = torch.unique(s_lbl)
        protos  = torch.stack([z_sup[s_lbl==c].mean(0) for c in classes])
        return -torch.cdist(z_qry, protos)

model = ProtoNet().to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

# ---------------------------------------------------------------------
# evaluation loop
accs = []
with torch.no_grad():
    for s_img, s_lbl, q_img, q_lbl in test_loader:
        # 🔑 remove extra batch-dim (1, 25, C, H, W) → (25, C, H, W)
        s_img, s_lbl = s_img.squeeze(0).to(device), s_lbl.squeeze(0).to(device)
        q_img, q_lbl = q_img.squeeze(0).to(device), q_lbl.squeeze(0).to(device)

        logits = model(s_img, s_lbl, q_img)
        preds  = logits.argmax(1)
        accs.append((preds == q_lbl).float().mean().item())

print(f"{args.n_way}-way {args.k_shot}-shot accuracy on {args.episodes} episodes: "
      f"{np.mean(accs):.3f}")
