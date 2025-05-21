import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
import os

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
train_loader, val_loader, test_loader = get_dataloaders()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

best_val_acc = 0.0
patience = 5
epochs_no_improve = 0

writer = SummaryWriter(log_dir='runs/ProtoPNet_Training_v2')

for epoch in range(30):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100. * val_correct / val_total
    scheduler.step(val_epoch_loss)

    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%")
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    writer.add_scalar("Loss/val", val_epoch_loss, epoch)
    writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)

    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f"✅ Best model saved at epoch {epoch+1} with Val Acc: {val_epoch_acc:.2f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

writer.close()
