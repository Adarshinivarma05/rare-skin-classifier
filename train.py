import torch, numpy as np, torch.nn as nn, torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import train_epoch, eval_epoch
from utils.visualizer import plot_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data
train_loader, val_loader, _, flat_labels = get_loaders(batch_size=32)

# Class weights
cls_w = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(flat_labels),
    y=flat_labels
)
cls_w = torch.tensor(cls_w, dtype=torch.float32).to(device)

# Model
model = ProtoPNet().to(device)

# Loss / Optimizer / Scheduler
criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6)
scaler    = torch.cuda.amp.GradScaler()

# Training loop
history, best_f1, patience, best_epoch = {k: [] for k in
    ["train_loss","val_loss","val_acc","val_f1"]}, 0, 6, 0

for epoch in range(1, 40):
    tl, ta, tf = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
    vl, va, vf = eval_epoch(model, val_loader, criterion, device)
    scheduler.step(epoch)
    history["train_loss"].append(tl); history["val_loss"].append(vl)
    history["val_acc"].append(va);    history["val_f1"].append(vf)

    print(f"Epoch {epoch:02d} | TL {tl:.3f} | VL {vl:.3f} | "
          f"VA {va*100:.1f}% | F1 {vf:.3f}")

    if vf > best_f1:
        torch.save(model.state_dict(), "best_model.pth")
        best_f1, best_epoch = vf, epoch
    elif epoch - best_epoch >= patience:
        print("Early stopping"); break

plot_history(history)
print("Best Val F1:", best_f1)
