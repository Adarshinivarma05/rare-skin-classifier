import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet(num_prototypes_per_class=10).to(device)

train_loader, val_loader, test_loader = get_dataloaders()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

writer = SummaryWriter(log_dir='runs/ProtoPNet_Training')

# Checkpointing setup
best_val_acc = 0.0
patience = 7
counter = 0
checkpoint_path = 'checkpoints/best_model.pth'
os.makedirs("checkpoints", exist_ok=True)

# Training
for epoch in range(50):
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

    print(f"Epoch {epoch+1}/50 | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    writer.add_scalar("Loss/val", val_epoch_loss, epoch)
    writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)

    # Save best model
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), checkpoint_path)
        print("✅ Best model saved.")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

writer.close()
print(f"🏁 Final Best Val Accuracy: {best_val_acc:.2f}%")
