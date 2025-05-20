import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.protopnet_skin_classifier import ProtoPSkinClassifier
from utils.data_loader import get_dataloaders
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 🧠 Model
    model = ProtoPSkinClassifier(num_prototypes=70, num_classes=7).to(device)

    # 📦 Data
    train_loader, val_loader, _ = get_dataloaders()

    # ⚙️ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 🛑 Early stopping
    early_stop_patience = 5
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # 📁 Checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    print("🔁 Starting training...")
    for epoch in range(1, 51):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()  # 🔧 FIXED

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
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze().long()  # 🔧 FIXED

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = 100 * correct / total

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/50 | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "checkpoints/best_model.pth")
            print("✅ Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
