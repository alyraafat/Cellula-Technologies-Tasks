import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple
import torch
import torch.utils
import torch.utils.data

def show_sat_img(img_path: str, way: str='channel'):
    assert way in ['channel', 'mean', 'rgb', 'none'], 'way should be either channel or mean'
    tif_image = tiff.imread(img_path)
    if way == 'channel':
        rows = 3
        cols = 4
        plt.figure(figsize=(15, 10))
        for i in range(tif_image.shape[-1]):
            curr_channel = tif_image[:, :, i]
            plt.subplot(rows, cols, i + 1)
            plt.title(f'Channel {i}')
            plt.imshow(curr_channel)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    elif way == 'mean':
        tif_image_avg = np.mean(tif_image, axis=-1)
        plt.title('Mean of channels')
        plt.imshow(tif_image_avg)
        plt.axis('off')
        plt.show()
    elif way == 'rgb':
        tif_image_copy = tif_image.copy().astype(np.float32)
        for i in range(tif_image_copy.shape[-1]):
            tif_image_copy[:,:,i] = (tif_image_copy[:,:,i] - tif_image_copy[:,:,i].min()) / (tif_image_copy[:,:,i].max() - tif_image_copy[:,:,i].min())
        plt.title('RGB')
        plt.imshow(tif_image_copy[:,:,[3, 2, 1]])
        plt.axis('off')
        plt.show()
    return tif_image



def show_labels(img_path: str):
    label_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    plt.imshow(label_img, cmap='gray')
    plt.axis('off')
    plt.show()
    return label_img

def get_max_min_channel(images_dir: str):
    max_channels = np.zeros(12)
    min_channels = np.zeros(12)
    for img_path in os.listdir(images_dir):
        img = tiff.imread(os.path.join(images_dir, img_path))
        for i in range(img.shape[-1]):
            max_channels[i] = max(max_channels[i], np.max(img[:, :, i]))
            min_channels[i] = min(min_channels[i], np.min(img[:, :, i]))
    return max_channels, min_channels


def get_mean_and_std(images_dir: str, images_paths: str, output_path: str='./mean_std.json'):
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            mean_std = json.load(f)
        return mean_std['mean'], mean_std['std']
    
    channels = tiff.imread(os.path.join(images_dir,images_paths[0])).shape[-1]
    channel_sums = np.zeros(channels, dtype=np.float64)
    channel_squared_sums = np.zeros(channels, dtype=np.float64)
    num_pixels = 0
    for img_path in images_paths:
        whole_path = os.path.join(images_dir, img_path)
        img = tiff.imread(whole_path).astype(np.float64) 
        num_pixels += img.shape[0] * img.shape[1]
        for i in range(channels):
            channel_sums[i] += np.sum(img[:, :, i])
            channel_squared_sums[i] += np.sum(np.square(img[:, :, i]))

    means = channel_sums / num_pixels
    sample_variance = (channel_squared_sums - (np.square(channel_sums) / num_pixels)) / (num_pixels - 1)
    stds = np.sqrt(sample_variance)
    mean_std = {'mean': means.tolist(), 'std': stds.tolist()}
    with open(output_path, 'w') as f:
        json.dump(mean_std, f)
    
    return means, stds



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



def load_state(obj: torch.nn.Module, checkpoint_path: str, device: str = '') -> torch.nn.Module:
    """
    Load a state_dict into a PyTorch object (e.g., model, optimizer) from a checkpoint file.

    Parameters:
    obj (torch.nn.Module or torch.optim.Optimizer): The object to load the state_dict into (e.g., model, optimizer).
    checkpoint_path (str): Path to the checkpoint file containing the state_dict.
    device (str): The device to map the loaded state_dict to (e.g., 'cpu' or 'cuda').

    Returns:
    obj: The object with the loaded state_dict.
    """
    if device == "":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Load the state_dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load the state_dict into the object
    obj.load_state_dict(state_dict)

    return obj


def visualize_predictions(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset, num_images: int=5):
    """
    Visualizes the predictions of a model on a few test images.

    Parameters:
    model (torch.nn.Module): The model to visualize predictions for.
    test_dataset (torch.utils.data.Dataset): The test dataset.
    num_images (int): Number of test images to visualize.
    """
    mean_std_path = './mean_std.json'
    with open(mean_std_path, 'r') as f:
            mean_std = json.load(f)
    mean = mean_std['mean']
    std = mean_std['std']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model.to(device)
    model.eval()
    for i in range(num_images):
        image, mask = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = model(image)
        pred_mask = torch.sigmoid(pred_mask) > 0.5
        pred_mask = pred_mask.squeeze().cpu().numpy()
        img_view = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        for i in [1,2,3]:
            img_view[:,:,i] = img_view[:,:,i] * std[i] + mean[i]
            # print(img_view[:,:,i].min(), img_view[:,:,i].max())
            img_view[:,:,i] = (img_view[:,:,i] - img_view[:,:,i].min()) / (img_view[:,:,i].max() - img_view[:,:,i].min())
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_view[:,:,[3,2,1]], cmap='gray')
        plt.axis('off')
        plt.title('Image')
        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth')
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.title('Prediction')
        plt.show()






