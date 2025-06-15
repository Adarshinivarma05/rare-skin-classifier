import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

class FocalLoss(nn.Module):
    """
    CE‑based focal loss to focus learning on harder / minority samples.
    gamma: 2.0 is standard; higher = more focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return torch.mean((1 - pt) ** self.gamma * ce)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)
        out = model(x)
        loss_sum += criterion(out, y).item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    loss = loss_sum / len(loader)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="weighted")
    return loss, acc, f1


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, preds, labels = 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out  = model(x)
            loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()
        preds.extend(out.argmax(1).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    loss = loss_sum / len(loader)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="weighted")
    return loss, acc, f1

