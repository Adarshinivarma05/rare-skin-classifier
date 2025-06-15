import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders
from utils.train_utils import eval_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ProtoPNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))

_, _, test_loader, _ = get_loaders(batch_size=32)
criterion = torch.nn.CrossEntropyLoss()
loss, acc, f1 = eval_epoch(model, test_loader, criterion, device)
print(f"Test Loss {loss:.3f} | Acc {acc*100:.2f}% | F1 {f1:.3f}")
