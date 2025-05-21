import torch
import torch.nn as nn
import torch.optim as optim
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)
    model = ProtoPNet().to(device)

    # Compute class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    best_val_acc = 0.0
    patience = 8
    trigger = 0

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/50 | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ Best model saved.")
        else:
            trigger += 1
            if trigger >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    train()
