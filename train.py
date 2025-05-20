import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data_loader import get_dataloaders
from models.protopnet_skin_classifier import ProtoPSkinClassifier
import os

# Training settings
num_epochs = 50
batch_size = 64
learning_rate = 0.001
patience = 5  # for early stopping
checkpoint_path = 'best_model.pth'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

# Model
model = ProtoPSkinClassifier(num_prototypes=70, num_classes=7).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping
best_val_loss = float('inf')
best_val_acc = 0.0
early_stop_counter = 0

print("🔁 Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Fix label shape
        labels = labels.squeeze()
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= total
    train_acc = 100.0 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Fix label shape
            labels = labels.squeeze()
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total

    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Checkpoint saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        print("✅ Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

    scheduler.step(val_loss)

print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.2f}%")
