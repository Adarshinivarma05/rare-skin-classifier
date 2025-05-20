import torch
import torch.nn as nn
from utils.data_loader import get_data_loaders
from models.protopnet_skin_classifier import ProtoPSkinClassifier

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = get_data_loaders()
    num_classes = len(class_names)

    model = ProtoPSkinClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()
