import matplotlib.pyplot as plt

def plot_metrics(history: dict, filename='metrics.png'):
    """Save a four‑panel training curve image."""
    titles = ['Train Loss', 'Val Loss', 'Val Accuracy', 'Val F1']
    keys   = ['train_loss', 'val_loss', 'val_acc', 'val_f1']

    plt.figure(figsize=(14, 4))
    for i, (k, t) in enumerate(zip(keys, titles)):
        plt.subplot(1, 4, i + 1)
        plt.plot(history[k])
        plt.title(t)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

