import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch.utils
import torch.utils.data
from tqdm import tqdm
from .metrics import calculate_metrics
from typing import List, Tuple

def evaluate(
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn, 
        class_names: List[str]=None, 
        device: str="", 
        is_testing: bool=False
    ) -> Tuple[float]: 

    if device == "":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate average loss
    avg_loss = running_loss / len(dataloader)

    # Calculate evaluation metrics
    accuracy, precision, recall, f1, dice_coeff, iou_val = calculate_metrics(np.array(all_labels), np.array(all_predictions))

    if is_testing:
        all_labels = np.array(all_labels).flatten()
        all_predictions = np.array(all_predictions).flatten()
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        class_report = classification_report(all_labels, all_predictions, target_names=class_names)
        print("Classification Report:")
        print(class_report)

        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Dice Coefficient: {dice_coeff:.4f}')
        print(f'Intersection over Union: {iou_val:.4f}')

    return avg_loss, accuracy, precision, recall, f1, dice_coeff, iou_val
