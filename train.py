# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import calculate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

train_loader, val_loader, _ = get_dataloaders()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

best_val_f1 = 0

for epoch in range(1, 21):
   model.train()
   total_loss = 0

   for images, labels in train_loader:
       images, labels = images.to(device), labels.squeeze().long().to(device)

       optimizer.zero_grad()
       outputs = model(images)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
       total_loss += loss.item()

   val_acc, val_f1 = calculate_metrics(model, val_loader, device)
   avg_loss = total_loss / len(train_loader)
   print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

   if val_f1 > best_val_f1:
       torch.save(model.state_dict(), "best_model.pth")
       best_val_f1 = val_f1


