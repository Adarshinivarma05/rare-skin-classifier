import torch
import torch.nn as nn
from models.protopnet_skin_classifier import ProtoPSkinClassifier
from utils.data_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    # 🧠 Load model
    model = ProtoPSkinClassifier(num_prototypes=70, num_classes=7).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    # 📦 Load test data
    _, _, test_loader = get_dataloaders()

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()  # 🔧 FIXED

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= total
    test_acc = 100 * correct / total

    print(f"✅ Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    test()
