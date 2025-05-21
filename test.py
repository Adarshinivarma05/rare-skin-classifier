import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    _, _, test_loader = get_dataloaders(batch_size=64)
    model = ProtoPNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    correct = 0
    total = 0

    tta_transforms = [
        transforms.Compose([]),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
    ]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits_sum = torch.zeros((images.size(0), 7)).to(device)

            for tta in tta_transforms:
                augmented = torch.stack([tta(img.cpu()) for img in images]).to(device)
                logits_sum += model(augmented)

            predictions = logits_sum.argmax(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Test Accuracy: {acc:.2%}")

if __name__ == "__main__":
    test()
