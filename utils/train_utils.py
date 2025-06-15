import torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def _loss_mixup(outputs, y_a, y_b, lam, criterion):
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

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
        preds_all.extend(outputs.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="weighted")
    )


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds_all.extend(outputs.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="weighted")
    )
