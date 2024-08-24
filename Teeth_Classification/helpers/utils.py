from sklearn.metrics import precision_score, recall_score, f1_score
import os
from typing import List, Tuple

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