import torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def epoch_step(model, loader, criterion, optimizer=None, scaler=None,
               device='cpu', mixup_fn=None):
    model.train() if optimizer else model.eval()
    run_loss, preds_all, labels_all = 0, [], []

    for x, y in loader:
        x, y = x.to(device), y.squeeze().long().to(device)

        # Optional MixUp only during training
        if optimizer and mixup_fn:
            x, y_a, y_b, lam = mixup_fn(x, y)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(x)
            if optimizer and mixup_fn:
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)

        if optimizer:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

        run_loss += loss.item()
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    return (run_loss/len(loader),
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average='weighted'))





