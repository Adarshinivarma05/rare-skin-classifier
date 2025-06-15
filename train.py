import torch
import torch.nn as nn
import torch.optim as optim
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import epoch_step
from utils.visualizer import plot_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

train_loader, val_loader, _ = get_dataloaders(batch_size=32)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler(device_type='cuda')

history = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_f1']}
best_f1 = 0

for epoch in range(1, 31):
    tr_loss, tr_acc, tr_f1 = epoch_step(model, train_loader, criterion, optimizer, scaler, device)
    val_loss, val_acc, val_f1 = epoch_step(model, val_loader, criterion, None, None, device)
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    print(f"Epoch {epoch:02d} | TL {tr_loss:.3f} | VL {val_loss:.3f} | VA {val_acc*100:.2f}% | F1 {val_f1:.3f}")

    if val_f1 > best_f1:
        torch.save(model.state_dict(), 'best_model.pth')
        best_f1 = val_f1

plot_metrics(history)




