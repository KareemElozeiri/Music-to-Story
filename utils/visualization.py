import os
import matplotlib.pyplot as plt


def plot_training_loss(train_losses, val_losses=None, save_dir=None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (list): Training loss values
        val_losses (list, optional): Validation loss values
        save_dir (str, optional): Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    else:
        plt.show()
