import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
model.load_state_dict(torch.load("protopnet_dermamnist.pth"))
model.eval()

_, _, test_loader = get_dataloaders()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
