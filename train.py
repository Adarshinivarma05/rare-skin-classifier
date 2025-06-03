import torch
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders
from utils.train_utils import compute_loss_weights, get_optimizer, get_scheduler
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

train_loader, _ = get_dataloaders(batch_size=64)
class_weights = compute_loss_weights(train_loader).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)

best_loss = float('inf')
patience, trigger = 5, 0

for epoch in range(30):
   model.train()
   running_loss = 0.0
   for images, labels in train_loader:
       images, labels = images.to(device), labels.squeeze().long().to(device)
       optimizer.zero_grad()
       outputs = model(images)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
       running_loss += loss.item()

   avg_loss = running_loss / len(train_loader)
   scheduler.step(avg_loss)

   print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

   # Early stopping and checkpoint
   if avg_loss < best_loss:
       best_loss = avg_loss
       trigger = 0
       torch.save(model.state_dict(), "protopnet_best.pth")
   else:
       trigger += 1
       if trigger >= patience:
           print("Early stopping triggered.")
           break


