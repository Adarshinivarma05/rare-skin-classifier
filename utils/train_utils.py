# utils/train_utils.py
import torch
from sklearn.metrics import f1_score

def calculate_metrics(model, dataloader, device):
   model.eval()
   correct = 0
   total = 0
   all_preds, all_labels = [], []

   with torch.no_grad():
       for images, labels in dataloader:
           images, labels = images.to(device), labels.squeeze().long().to(device)
           outputs = model(images)
           _, preds = torch.max(outputs, 1)
           correct += (preds == labels).sum().item()
           total += labels.size(0)
           all_preds.extend(preds.cpu().numpy())
           all_labels.extend(labels.cpu().numpy())

   acc = 100 * correct / total
   f1 = f1_score(all_labels, all_preds, average='weighted')
   return acc, f1

