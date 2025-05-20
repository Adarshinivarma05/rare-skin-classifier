import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and weights
model = ProtoPNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Get test loader
_, test_loader = get_dataloaders()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Accuracy Output
print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
