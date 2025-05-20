import torch
import torch.nn as nn
import torch.optim as optim
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import EarlyStopping, save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

train_loader, val_loader = get_dataloaders()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

early_stopping = EarlyStopping(patience=10, min_delta=0.001)

best_val_loss = float('inf')
for epoch in range(50):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = correct / total * 100

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total * 100
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/50 | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, 'best_model.pth')
        print("✅ Best model saved.")

    if early_stopping(val_loss):
        print("⏹️ Early stopping triggered.")
        break
