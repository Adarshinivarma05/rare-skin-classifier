# test.py
import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import calculate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

_, _, test_loader = get_dataloaders()

acc, f1 = calculate_metrics(model, test_loader, device)
print(f"Test Accuracy: {acc:.2f}% | F1 Score: {f1:.4f}")

