# train.py
import torch, numpy as np
import torch.nn as nn, torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from utils.data_loader import load_dermamnist_dataset
from utils.train_utils import train_epoch, evaluate_model
from models.protopnet_skin_classifier import ProtoPNet

# ─── Device ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ─── Data ──────────────────────────────────────────────────────────────────
train_loader, val_loader, _, num_classes, flat_labels = load_dermamnist_dataset(batch_size=32)

# ─── Class‑balanced weights ───────────────────────────────────────────────
cls_w = compute_class_weight(class_weight="balanced",
                             classes=np.unique(flat_labels),
                             y=flat_labels)
cls_w = torch.tensor(cls_w, dtype=torch.float32).to(device)

# ─── Model / loss / optimizer ─────────────────────────────────────────────
model = ProtoPNet(num_prototypes=60, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler    = torch.cuda.amp.GradScaler()

# ─── Training loop ────────────────────────────────────────────────────────
num_epochs = 25
best_f1 = 0
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion,
                                         optimizer, device, scaler)
    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch:02d} | TL {tr_loss:.3f} | VL {val_loss:.3f} | "
          f"VA {val_acc*100:.1f}% | F1 {val_f1:.3f}")

    if val_f1 > best_f1:
        torch.save(model.state_dict(), "best_model.pth")
        best_f1 = val_f1

print("Best Validation F1:", best_f1)





