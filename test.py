import torch
from models.protopnet_skin_classifier import ProtoPNet
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

info = INFO['dermamnist']
DataClass = DermaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_dataset = DataClass(split='test', transform=data_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet(num_prototypes=30, num_classes=7).to(device)
model.load_state_dict(torch.load('protopnet_skin.pt'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).squeeze()
        outputs, _ = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")

# Your actual code here
