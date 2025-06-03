import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
model.load_state_dict(torch.load("protopnet_best.pth"))
model.eval()

_, test_loader = get_dataloaders()

correct, total, test_loss = 0, 0, 0.0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
   for images, labels in test_loader:
       images, labels = images.to(device), labels.squeeze().long().to(device)
       outputs = model(images)
       loss = criterion(outputs, labels)
       test_loss += loss.item()
       _, preds = torch.max(outputs, 1)
       correct += (preds == labels).sum().item()
       total += labels.size(0)

avg_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
