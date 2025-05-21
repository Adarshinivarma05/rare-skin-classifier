import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import get_dataloaders
from models.protopnet_skin_classifier import ProtoPNet

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()

    model = ProtoPNet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f"✅ Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(all_labels, all_preds))
    print("\n🧾 Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    test()
