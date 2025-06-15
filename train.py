import torch, numpy as np
import torch.nn as nn, torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from medmnist import DermaMNIST
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import epoch_step
from utils.visualizer import plot_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

# Freeze backbone except layer4
for n,p in model.backbone.named_parameters():
    if not n.startswith('layer4'): p.requires_grad = False

train_loader, val_loader, _, mixup_fn = get_dataloaders(batch_size=32)

# class‑balanced weights
labels = np.array(DermaMNIST(split='train', download=True).labels).squeeze()
w = compute_class_weight('balanced', np.unique(labels), labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device),
                                label_smoothing=0.1)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
scaler = torch.cuda.amp.GradScaler()

hist = {k:[] for k in ['train_loss','val_loss','val_acc','val_f1']}
best_f1, best_e, patience = 0,0,6

for ep in range(1,40):
    tl, ta, tf = epoch_step(model, train_loader, criterion,
                            optimizer, scaler, device, mixup_fn)
    vl, va, vf = epoch_step(model, val_loader, criterion,
                            device=device)  # eval, no mixup
    scheduler.step(ep+vl)

    hist['train_loss'].append(tl); hist['val_loss'].append(vl)
    hist['val_acc'].append(va);   hist['val_f1'].append(vf)
    print(f"E{ep:02d} TL{tl:.3f} VL{vl:.3f} VA{va*100:.1f}% F1{vf:.3f}")

    if vf>best_f1: torch.save(model.state_dict(),'best_model.pth'); best_f1,vf
    elif ep-best_e>=patience: print("Early stop"); break
    best_e = ep if vf>best_f1 else best_e

plot_metrics(hist)




