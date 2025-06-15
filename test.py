import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import epoch_step

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device).eval()
model.load_state_dict(torch.load('best_model.pth'))

_, _, test_loader, _ = get_dataloaders(batch_size=32)
loss, acc, f1 = epoch_step(model, test_loader,
                           torch.nn.CrossEntropyLoss(), device=device)
print(f"Test Loss {loss:.3f} | Acc {acc*100:.2f}% | F1 {f1:.3f}")



