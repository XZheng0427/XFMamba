import os
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoints-mura'):
        """
        Args:
            patience (int): How long to wait after last validation loss improvement.
            verbose (bool): If True, prints messages about early stopping actions.
            delta (float): Minimum change in the monitored quantity to qualify as improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Counts epochs with no improvement
        self.best_score = None
        self.early_stop = False  # Flag to indicate if early stopping should occur
        self.val_loss_min = float('inf')  # Initialize to infinity
        self.delta = delta
        self.path = path  # Where to save the model

    def __call__(self, val_loss, model, epoch):
        score = -val_loss  # We want to maximize the negative loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)  # Save initial model
        elif score < self.best_score + self.delta:
            self.counter += 1  # No improvement
            self.save_checkpoint(val_loss, model, epoch)  # Save new best model
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping activated')
                self.early_stop = True
        else:
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model, epoch)  # Save new best model
            self.counter = 0  # Reset counter

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        save_path = os.path.join(self.path, f"model_epoch_{epoch + 1}.pth")
        os.makedirs(self.path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss  # Update the minimum validation loss
 