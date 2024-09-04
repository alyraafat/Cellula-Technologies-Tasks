from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np
from typing import Union

def dice_coefficient(y_true: Union[np.ndarray,torch.Tensor], y_pred: Union[np.ndarray,torch.Tensor], smooth: float=1.0, flat_tensors: bool=True) -> float:
    if flat_tensors:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    sum_of_positives = y_true.sum() + y_pred.sum()
    return (2. * intersection + smooth) / (sum_of_positives + smooth)

def iou(y_true: np.ndarray, y_pred: np.ndarray, smooth: float=1.0, flat_tensors: bool=True) -> float:
    if flat_tensors:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float]:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = 100 * (y_true == y_pred).sum() / len(y_true)
    dice_coeff = dice_coefficient(y_true, y_pred, flat_tensors=False)
    iou_val = iou(y_true, y_pred, flat_tensors=False)
    return accuracy, precision, recall, f1, dice_coeff, iou_val