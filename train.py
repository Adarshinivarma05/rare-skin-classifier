# train.py   (overwrite or edit imports + setup)
import torch, numpy as np
import torch.nn as nn, torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import train_epoch, eval_epoch, FocalLoss
from utils.visualizer import plot_history

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)

# ─── Data
train_loader, val_loader, _, flat_labels = get_loaders(batch_size=32)

# ─── Class‑balanced weights
cls_w = torch.tensor(
    compute_class_weight(class_weight="balanced",
                         classes=np.unique(flat_labels),
                         y=flat_labels),
    dtype=torch.float32, device=DEVICE)

# ─── Model
model = ProtoPNet().to(DEVICE)

# ─── Phase‑1: freeze backbone except layer4
for n, p in model.backbone.named_parameters():
    if not n.startswith("layer4"):
        p.requires_grad = False

criterion = FocalLoss(weight=cls_w, gamma=2.0).to(DEVICE)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
scaler    = torch.cuda.amp.GradScaler()

history = {k: [] for k in ["train_loss", "val_loss", "val_acc", "val_f1"]}
best_f1, patience, best_ep = 0, 6, 0

# ─── Training Loop
for epoch in range(1, 40):
    # Un‑freeze entire backbone after 6 warm‑up epochs & drop LR
    if epoch == 7:
        for p in model.backbone.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
        print("→ Un‑froze backbone, lowered LR to 5e‑5")

    tl, ta, tf = train_epoch(model, train_loader, criterion,
                             optimizer, scaler, DEVICE)
    vl, va, vf = eval_epoch(model, val_loader,  criterion, DEVICE)
    scheduler.step(epoch)

    for k, v in zip(history.keys(), [tl, vl, va, vf]): history[k].append(v)
    print(f"E{epoch:02d} TL {tl:.3f} VL {vl:.3f} VA {va*100:.1f}% F1 {vf:.3f}")

    if vf > best_f1:
        torch.save(model.state_dict(), "best_model.pth")
        best_f1, best_ep = vf, epoch
    elif epoch - best_ep >= patience:
        print("Early stopping at epoch", epoch)
        break

plot_history(history)
print("Best Val F1:", best_f1)
