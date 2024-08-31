import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os

def show_sat_img(img_path: str, way: str='channel'):
    assert way in ['channel', 'mean'], 'way should be either channel or mean'
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
    return tif_image



def show_labels(img_path: str):
    label_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    plt.imshow(label_img, cmap='gray')
    plt.axis('off')
    plt.show()
    return label_img


def get_mean_and_std(images_dir: str, images_paths: str, output_path: str='../mean_std.json'):
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
