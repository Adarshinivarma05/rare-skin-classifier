import torch
from sklearn.metrics import accuracy_score, f1_score


def epoch_step(model, loader, criterion, optimizer=None, scaler=None, device='cpu'):
    """
    Run one epoch. If `optimizer` is None, runs in eval mode.
    Supports AMP mixed precision through `scaler`.
    Returns loss, accuracy, f1.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss_avg = running_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='weighted')
    return loss_avg, acc, f1
