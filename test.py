import torch
from utils.data_loader import get_dataloaders
from models.protopnet_skin_classifier import ProtoPSkinClassifier
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
_, _, test_loader = get_dataloaders(batch_size=64)

# Load model
model = ProtoPSkinClassifier(num_prototypes=70, num_classes=7)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Fix label shape
        labels = labels.squeeze()
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = 100.0 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
