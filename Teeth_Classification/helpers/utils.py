from sklearn.metrics import precision_score, recall_score, f1_score
import os
from typing import List, Tuple
import matplotlib.pyplot as plt


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float]:
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = 100 * (y_true == y_pred).sum() / len(y_true)
    return accuracy, precision, recall, f1


def show_counts(train_dir: str) -> Tuple[dict, dict, dict]:
    class_counts = {}
    total_count = 0
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)

        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            class_counts[class_name] = num_images
            total_count += num_images
    class_weights = {}
    sum_of_weights = 0
    for class_name, count in class_counts.items():
        print(f"Class '{class_name}' contains {count} images.")
        class_weights[class_name] = total_count / count
        sum_of_weights += total_count / count
    class_weights_norm = {class_name: weight / sum_of_weights for class_name, weight in class_weights.items()}

    return class_weights, class_weights_norm, class_counts


import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    history (dict): A dictionary containing 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
                    Each key should map to a list of values (one per epoch).
    """

    epochs = range(1, len(history['train_loss']) + 1)

    # Plotting Loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


