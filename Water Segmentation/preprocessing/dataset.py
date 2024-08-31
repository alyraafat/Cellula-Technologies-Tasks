from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import tifffile as tiff
from torchvision import transforms
import json
import random
import torch
from typing import List, Tuple, Union

class WaterSegDataset(Dataset):
    def __init__(self, 
                 images_dir: str, 
                 masks_dir: str, 
                 zipped_paths: List[Union[List[str],Tuple[str,str]]], 
                 type: str='train', 
                 mean_std_path: str='../mean_std.json'):
        super(WaterSegDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.images_paths = [img_path for img_path, _ in zipped_paths]
        self.masks_paths = [mask_path for _, mask_path in zipped_paths]

        # # clean masks_paths by removing the paths that contain '_' in their names
        # self.masks_paths = [mask_path for mask_path in self.masks_paths if '_' not in mask_path]

        # # sort the images and masks paths
        # self.images_paths = sorted(self.images_paths, key=lambda x: int(x.split('.')[0]))
        # self.masks_paths = sorted(self.masks_paths, key=lambda x: int(x.split('.')[0]))

        # prepare transforms
        with open(mean_std_path, 'r') as f:
            mean_std = json.load(f)
        self.mean = mean_std['mean']
        self.std = mean_std['std']
        self.transforms = SegmentationTransforms(mean=self.mean, std=self.std, augment=(type == 'train'))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images_paths[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_paths[idx])
        print(f'Image path: {image_path}, Mask path: {mask_path}')
        
        image = tiff.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        image, mask = self.transforms(image, mask)

        return image, mask


class SegmentationTransforms:
    def __init__(self, mean, std, augment=True):
        self.mean = mean
        self.std = std
        self.augment = augment

    def __call__(self, image, mask):
        if self.augment:
            image, mask = self.random_flip(image, mask)
            image, mask = self.random_rotate(image, mask)

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        image = transforms.Normalize(mean=self.mean, std=self.std)(image)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
        
        return image, mask

    def random_flip(self, image, mask):
        # Random Horizontal Flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random Vertical Flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return image, mask

    def random_rotate(self, image, mask, degrees=(-45, 45)):
        angle = random.uniform(*degrees)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return image, mask
