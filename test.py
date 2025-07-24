import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import eval_epoch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='proto_resnet50.pth')  # Use this model
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProtoPNet().to(device)

# ❌ Remove this broken line
# model.load_state_dict(torch.load("best_model.pth"))

# ✅ Load model safely with correct filename
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

_, _, test_loader, _ = get_loaders(batch_size=32)
criterion = torch.nn.CrossEntropyLoss()

loss, acc, f1 = eval_epoch(model, test_loader, criterion, device)
print(f"Test Loss {loss:.3f} | Acc {acc*100:.2f}% | F1 {f1:.3f}")
