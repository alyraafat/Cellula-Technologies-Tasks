import torch

class SaveBestModel:
    def __init__(self, best_val_loss: float=float('inf'), save_path: str='../models_weights/best_model.pth'):
        self.best_val_loss = best_val_loss
        self.save_path = save_path

    def __call__(self, model: torch.nn.Module, current_val_loss: float, prev_best_min_loss: float):
        if current_val_loss < self.best_val_loss and current_val_loss < prev_best_min_loss:
            self.best_val_loss = current_val_loss
            torch.save(model, self.save_path)
            print(f'Best model saved with Val Loss: {self.best_val_loss:.4f}')


   