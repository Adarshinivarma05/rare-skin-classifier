import torch
from sklearn.metrics import f1_score, accuracy_score

def epoch_step(model, loader, criterion, optimizer=None, scaler=None, device='cpu'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):

            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss_avg = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return loss_avg, acc, f1




