print("
import torch
import torch.nn as nn
import torch.optim as optim
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet().to(device)

train_loader, _ = get_dataloaders()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
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

    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "protopnet_dermamnist.pth")
#SPOORTHI
      ")
