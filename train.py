import torch
from torch import nn, optim
from models.protopnet_skin_classifier import ProtoPNet
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.visualizer import plot_loss_curve

info = INFO['dermamnist']
DataClass = DermaMNIST

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=data_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoPNet(num_prototypes=30, num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10
losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).squeeze()  # labels shape fix
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), 'protopnet_skin.pt')
plot_loss_curve(losses)


# Your actual code here
