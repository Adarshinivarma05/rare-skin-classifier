# utils/train_utils.py
import torch
from sklearn.metrics import accuracy_score, f1_score

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.squeeze().long().to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.squeeze().long().to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1

