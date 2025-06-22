
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from medmnist import DermaMNIST
from utils.episode_loader import get_episode_loader

# ----------------------------- CLI args -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_way",      type=int,   default=3,      help="classes per episode")
parser.add_argument("--k_shot",     type=int,   default=1,      help="support images/class")
parser.add_argument("--q",          type=int,   default=1,      help="query images/class")
parser.add_argument("--episodes",   type=int,   default=1000,   help="episodes per epoch")
parser.add_argument("--epochs",     type=int,   default=20,     help="training epochs")
parser.add_argument("--lr",         type=float, default=1e-4,   help="learning rate")
parser.add_argument("--save",       type=str,   default="proto_resnet50.pth",
                    help="checkpoint filename")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- dataset -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1-channel to 3-channel
    transforms.Normalize([.5, .5, .5], [.5, .5, .5])
])
train_ds = DermaMNIST(split="train", transform=transform, download=True)
val_ds   = DermaMNIST(split="val",   transform=transform, download=True)

train_loader = get_episode_loader(train_ds, args.n_way, args.k_shot, args.q,
                                  args.episodes, shuffle=True)
val_loader   = get_episode_loader(val_ds,   args.n_way, args.k_shot, args.q,
                                  episodes_per_epoch=300, shuffle=False)

# ----------------------------- model -----------------------------
class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()           # 2048-d embedding
        self.encoder = backbone

    def forward(self, s_img, s_lbl, q_img):
        z_support = self.encoder(s_img)       # (N*K, 2048)
        z_query   = self.encoder(q_img)       # (N*Q, 2048)
        classes   = torch.unique(s_lbl)
        protos    = torch.stack([z_support[s_lbl==c].mean(0) for c in classes])
        dists     = torch.cdist(z_query, protos)          # (N*Q, N)
        return -dists                                     # logits

model  = ProtoNet().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler()

# ----------------------------- loops -----------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_list, acc_list = [], []

    for s_img, s_lbl, q_img, q_lbl in loader:
        s_img, s_lbl = s_img.squeeze(0).to(device), s_lbl.squeeze(0).to(device)
        q_img, q_lbl = q_img.squeeze(0).to(device), q_lbl.squeeze(0).to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(s_img, s_lbl, q_img)
            loss   = F.cross_entropy(logits, q_lbl)

        if train:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        preds = logits.argmax(1)
        acc   = (preds == q_lbl).float().mean()
        loss_list.append(loss.item()); acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)

# ----------------------------- training -----------------------------
for ep in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    print(f"Epoch {ep:02d} | train acc {tr_acc:.3f} | val acc {vl_acc:.3f}")

torch.save(model.state_dict(), args.save)
print(f"✔️  model saved to {args.save}")
