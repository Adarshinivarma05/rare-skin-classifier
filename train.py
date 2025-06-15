import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from medmnist import DermaMNIST
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import epoch_step
from utils.visualizer import plot_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = ProtoPNet().to(device)

# ------------- Data -------------
train_loader, val_loader, _ = get_dataloaders()

# ------------- Class weights -------------
labels = np.array(DermaMNIST(split='train', download=True).labels).squeeze()
cls_w  = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
cls_w  = torch.tensor(cls_w, dtype=torch.float).to(device)

# ------------- Criterion / Optimizer / Scheduler -------------
criterion  = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
optimizer  = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
scaler     = torch.cuda.amp.GradScaler()

# ------------- Training Loop -------------
history, best_f1, best_epoch, patience = {k: [] for k in
    ['train_loss','val_loss','val_acc','val_f1']}, 0.0, 0, 7

for epoch in range(1, 50):  # max 50 epochs, early stop likely sooner
    tr_loss, tr_acc, tr_f1 = epoch_step(model, train_loader, criterion, optimizer, scaler, device)
    val_loss, val_acc, val_f1 = epoch_step(model, val_loader, criterion, None, None, device)
    scheduler.step(epoch + val_loss)   # scheduler tick

    # save history
    history['train_loss'].append(tr_loss)
    history['val_loss'  ].append(val_loss)
    history['val_acc'   ].append(val_acc)
    history['val_f1'    ].append(val_f1)

    print(f"Epoch {epoch:02d} | TL {tr_loss:.3f} | VL {val_loss:.3f} | "
          f"VA {val_acc*100:.1f}% | F1 {val_f1:.3f}")

    # Early stopping check
    if val_f1 > best_f1:
        best_f1, best_epoch = val_f1, epoch
        torch.save(model.state_dict(), 'best_model.pth')
    elif epoch - best_epoch >= patience:
        print(f"Early stop @ {epoch}")
        break

# save curves
plot_metrics(history)
print(f"Best Val F1: {best_f1:.3f} at epoch {best_epoch}")

