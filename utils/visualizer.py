import matplotlib.pyplot as plt

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# Your actual code here
