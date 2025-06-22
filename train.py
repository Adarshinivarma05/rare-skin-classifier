"""
Episodic Prototypical Net training
– DermaMNIST 224×224, 5-way 5-shot (any N-way K-shot)
– ResNet-50 (ImageNet-pretrained) backbone
– Data augmentation, AMP, LR scheduler
"""

import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T, models
from medmnist import DermaMNIST
from utils.episode_loader import get_episode_loader

# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_way",      type=int, default=5,      help="classes per episode")
parser.add_argument("--k_shot",     type=int, default=5,      help="support images/class")
parser.add_argument("--q",          type=int, default=1,      help="query images/class")
parser.add_argument("--episodes",   type=int, default=3000,   help="episodes per epoch")
parser.add_argument("--epochs",     type=int, default=30,     help="training epochs")
parser.add_argument("--lr",         type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--save",       type=str, default="proto_resnet50.pth",
                    help="checkpoint file")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Data transforms  (augment + grayscale-to-RGB)
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3,1,1)),
    T.Normalize([.5, .5, .5], [.5, .5, .5])
])

train_ds = DermaMNIST(split="train", transform=transform, download=True)
val_ds   = DermaMNIST(split="val",   transform=transform, download=True)

train_loader = get_episode_loader(train_ds, args.n_way, args.k_shot, args.q,
                                  args.episodes, shuffle=True)
val_loader   = get_episode_loader(val_ds,   args.n_way, args.k_shot, args.q,
                                  episodes_per_epoch=600, shuffle=False)

# ------------------------------------------------------------------
class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V2")
        backbone.fc = nn.Identity()                # 2048-d embedding
        self.encoder = backbone

    def forward(self, s_img, s_lbl, q_img):
        z_sup  = self.encoder(s_img)               # (N*K, 2048)
        z_qry  = self.encoder(q_img)               # (N*Q, 2048)
        classes = torch.unique(s_lbl)
        protos  = torch.stack([z_sup[s_lbl==c].mean(0) for c in classes])
        dists   = torch.cdist(z_qry, protos)       # (N*Q, N)
        return -dists                              # logits

model  = ProtoNet().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.2)
scaler = torch.amp.GradScaler()

# ------------------------------------------------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    losses, accs = [], []

    for s_img, s_lbl, q_img, q_lbl in loader:
        # drop extra DataLoader dim → (25, C, H, W)
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
        losses.append(loss.item()); accs.append(acc.item())

    return np.mean(losses), np.mean(accs)

# ------------------------------------------------------------------
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    sched.step()

    print(f"Epoch {epoch:02d}/{args.epochs}  │  "
          f"train {tr_acc:.3f}  val {vl_acc:.3f}")

torch.save(model.state_dict(), args.save)
print(f"✔️  saved episodic model → {args.save}")
