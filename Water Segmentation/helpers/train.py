import torch
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
from .metrics import calculate_metrics
from .callbacks import SaveBestModel
from .evaluate import evaluate
from typing import Dict, Union, List

def train(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn, 
        optimizer: torch.optim, 
        num_epochs: int=10, 
        device: str="", 
        start_epoch: int=0, 
        scheduler: torch.optim.lr_scheduler=None, 
        new_lr: float=None,
        models_weights_path: str='../models_weights/best_model.pth',
        prev_best_min_loss: float=float('inf'),
        prev_history: dict=None
    ) -> Dict[str,Union[float, List[float]]]:

    if device == "":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.compile(model, backend="nvfuser")
    model = model.to(device)
    save_best_model = SaveBestModel(save_path=models_weights_path)
    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        train_accuracy, train_precision, train_recall, train_f1, train_dice_coeff, train_iou_val = calculate_metrics(np.array(all_labels), np.array(all_predictions))
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, '
              f'Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}, Dice Coefficient: {train_dice_coeff:.4f}, IOU: {train_iou_val:.4f}')

        train_accuracies.append(train_accuracy)
        # Validation phase
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_dice_coeff, val_iou_val = evaluate(val_loader, model, criterion)

        print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, '
              f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, Dice Coefficient: {val_dice_coeff:.4f}, IOU: {val_iou_val:.4f}')
        print("-"*50)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        # Save the best model using the callback
        save_best_model(model, avg_val_loss, prev_best_min_loss)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step(avg_val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

    print('Finished Training')
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        "prev_best_min_loss": save_best_model.best_val_loss
    }

    if prev_history is not None:
        for key in prev_history.keys():
            if key != 'prev_best_min_loss':
                history[key] =  prev_history[key] + history[key]

    return history