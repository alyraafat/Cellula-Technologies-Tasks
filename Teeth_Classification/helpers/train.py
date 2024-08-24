import torch
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
from .utils import calculate_metrics
from .callbacks import SaveBestModel
from .evaluate import evaluate

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
        models_weights_path: str='../models_weights/best_model.pth'
    ) -> None:

    if device == "":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.compile(model, backend="nvfuser")
    model = model.to(device)
    save_best_model = SaveBestModel(save_path=models_weights_path)
    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

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

            predicted = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)

        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(np.array(all_labels), np.array(all_predictions))
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, '
              f'Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}')

        # Validation phase
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(val_loader, model, criterion)

        print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, '
              f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')
        print("-"*50)

        # Save the best model using the callback
        save_best_model(model, avg_val_loss)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step(avg_val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)
