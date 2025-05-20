import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.protopnet_skin_classifier import ProtoPSkinClassifier
from utils.data_loader import get_data_loaders
import os

def train():
    # Hyperparameters
    data_dir = 'data/skin_images'
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    patience = 6
    num_prototypes = 70
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)

    # Model
    model = ProtoPSkinClassifier(num_classes=num_classes, num_prototypes=num_prototypes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training loop with early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = 'best_model.pt'

    print("🔁 Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

    print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    train()
