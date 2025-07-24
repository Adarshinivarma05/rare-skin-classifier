import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import train_epoch, eval_epoch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProtoPNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

train_loader, val_loader, _, _ = get_loaders(batch_size=16)

best_val_acc = 0.0
for epoch in range(1, 31):
    model.train()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

    model.eval()
    val_loss, val_acc, _ = eval_epoch(model, val_loader, criterion, device)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "proto_resnet50.pth")
