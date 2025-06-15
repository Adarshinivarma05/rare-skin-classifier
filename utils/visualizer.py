import matplotlib.pyplot as plt

def plot_metrics(history, filename='metrics.png'):
    plt.figure(figsize=(12, 4))
    for i, key in enumerate(['train_loss', 'val_loss', 'val_acc', 'val_f1']):
        plt.subplot(1, 4, i + 1)
        plt.plot(history[key])
        plt.title(key)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




