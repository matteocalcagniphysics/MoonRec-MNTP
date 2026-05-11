import torch
from torch.utils.data import Dataset
from scipy.ndimage import label
import numpy as np
import pandas as pd
import random
from pathlib import Path

class MoonTileDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, augment: bool = False):
        self.index_df = index_df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.index_df)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        # Horizontal Flip
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        # Vertical Flip
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        # Random Rotation (90, 180, 270)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        data = np.load(row['tile_path'])
        image = data['image'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)
        return torch.from_numpy(image), torch.from_numpy(mask)


class MoonTileTestDataset_RCNN(Dataset):
    """
    Dataset for Mask R-CNN inference.
    It only returns the image tensor, without the mask, since we will be using the model for inference and not training.
    The image tensor will be in the shape (C, H, W) and will be normalized to [0, 1].
    """

    def __init__(self, index_df: pd.DataFrame, augment: bool = False):
        self.index_df = index_df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.index_df)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        # Horizontal Flip
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        # Vertical Flip
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        # Random Rotation (90, 180, 270)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        data = np.load(row['tile_path'])

        # Load the image and normalize to [0, 1]
        image = data['image'].astype(np.float32)
        mask = data['mask'].astype(np.uint8)  # Better to keep mask as uint8 for inference
        if image.max() > 1.0:
            image /= 255.0
        
        # Augementation (if needed)
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Geometry estraction for Mask R-CNN
        boxes_coord = []
        valid_masks = []
        class_labels = []

        for class_idx in range(mask.shape[0]):
            # Extract the mask
            class_mask = mask[class_idx]

            # Skip if the channel is empty
            if class_mask.max() == 0:
                continue

            # Find all the pixels that aren't zero
            pixel = np.where(class_mask > 0)

            # We use 'label' from scipy.ndimage to find groups of pixels "physically" near
            # num_instances tells us how many groups it found  
            labeled_mask, num_instances = label(class_mask)
            
            for instance_id in range(1, num_instances + 1):
                # Create a binary mask for each instance
                m_inst = (labeled_mask == instance_id).astype(np.uint8)
                
                # Find all the pixels that are non-zero
                pos = np.where(m_inst > 0)
                
                if len(pos[0]) == 0:
                    continue
                
                # Find the coordinates of the boxes
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                # Check valid boxes
                if xmax > xmin and ymax > ymin:
                    boxes_coord.append([xmin, ymin, xmax, ymax])
                    valid_masks.append(m_inst)
                    # Save the class, add 1 because in R-CNNs the 0 is for the background
                    class_labels.append(class_idx + 1)
