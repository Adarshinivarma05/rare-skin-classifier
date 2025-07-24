import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from models.protopnet_skin_classifier import ProtoPNet
from utils.data_loader import get_loaders, get_few_shot_loaders
from utils.train_utils import train_epoch, eval_epoch, few_shot_train_epoch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_way', type=int, default=5, help="Number of classes per episode")
    parser.add_argument('--k_shot', type=int, default=1, help="Support examples per class")
    parser.add_argument('--q', type=int, default=5, help="Query examples per class")
    parser.add_argument('--episodes', type=int, default=100, help="Episodes per epoch (few-shot)")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save', type=str, default="proto_resnet50.pth", help="Save model path")
    
    # ✅ These enable special AI techniques
    parser.add_argument('--few_shot', action='store_true', help="Enable few-shot training mode")
    parser.add_argument('--use_proto', action='store_true', help="Enable ProtoPNet-like prototype layer usage")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoPNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # ✅ Use Few-Shot or Full Supervised Training
    if args.few_shot:
        train_loader, val_loader = get_few_shot_loaders(n_way=args.n_way, k_shot=args.k_shot, q=args.q, episodes=args.episodes)
    else:
        train_loader, val_loader, _, _ = get_loaders(batch_size=16)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.few_shot:
            train_loss, train_acc = few_shot_train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        else:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        model.eval()
        val_loss, val_acc, _ = eval_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save)
            print(f"✅ Best model saved to {args.save}")

if __name__ == "__main__":
    main()
