import os
import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import eval_epoch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='Path to model file')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProtoPNet().to(device)

# Figure out correct model path
if args.model_path is not None:
    model_path = args.model_path
else:
    # Dynamically find the path one directory above test.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "best_model.pth")

# Load model safely
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))

# Load data and evaluate
_, _, test_loader, _ = get_loaders(batch_size=32)
criterion = torch.nn.CrossEntropyLoss()
loss, acc, f1 = eval_epoch(model, test_loader, criterion, device)
print(f"Test Loss: {loss:.3f} | Accuracy: {acc*100:.2f}% | F1 Score: {f1:.3f}")
