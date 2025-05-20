import torch
from models.protopnet_skin_classifier import ProtoPSkinClassifier
from utils.data_loader import get_data_loaders

def test():
    # Setup
    data_dir = 'data/skin_images'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    _, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    num_prototypes = 70

    # Load model
    model = ProtoPSkinClassifier(num_classes=num_classes, num_prototypes=num_prototypes)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    test()
