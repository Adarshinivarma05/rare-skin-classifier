# utils/train_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class FocalLoss(nn.Module):
    """
    Cross‑entropy‑based focal loss.
    gamma > 1 focuses learning on harder / minority samples.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight, reduction="none")
        pt = torch.exp(-ce)           # prob of correct class
        return torch.mean((1 - pt) ** self.gamma * ce)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Validation / test loop: returns loss, acc, weighted‑F1."""
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="weighted")
    return loss, acc, f1


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """One training epoch with AMP."""
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss   = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="weighted")
    return loss, acc, f1

def few_shot_train_epoch(model, optimizer, loader, device, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        support_images, support_labels, query_images, query_labels = batch

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()

        # Combine support and query images to pass through ProtoPNet
        all_images = torch.cat([support_images, query_images], dim=0)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(all_images)

        # Extract only query outputs for loss calculation
        query_outputs = outputs[-len(query_labels):]
        loss = criterion(query_outputs, query_labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = query_outputs.max(1)
        correct += predicted.eq(query_labels).sum().item()
        total += query_labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy



