# train.py

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_loader import load_dermamnist_dataset
from utils.train_utils import train_epoch, evaluate_model
from models.protopnet_skin_classifier import ProtoPNet

from sklearn.utils.class_weight import compute_class_weight

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 1: Data loading
train_loader, val_loader, test_loader, num_classes, labels = load_dermamnist_dataset(batch_size=32)

# Step 2: Compute class weights for balanced loss
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Step 3: Model init
model = ProtoPNet(num_classes=num_classes).to(device)

# Step 4: Loss, optimizer, scaler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# Step 5: Training loop
num_epochs = 25
for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch:02d} | TL {train_loss:.3f} | VL {val_loss:.3f} | VA {val_acc:.2f}% | F1 {val_f1:.3f}")

# Optional: Save final model
torch.save(model.state_dict(), 'model_final.pth')




