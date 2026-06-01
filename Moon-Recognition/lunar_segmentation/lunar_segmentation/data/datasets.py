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

        # Load the image and normalize to [0, 1] if needed
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


        # Prepare the tensors for Mask R-CNN
        # "Unlucky" case: no valid boxes found, we need to return something for the model to work
        if len(boxes_coord) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros((0, mask.shape[1], mask.shape[2]), dtype=torch.uint8)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes_coord, dtype=torch.float32)
            masks_tensor = torch.tensor(np.array(valid_masks), dtype=torch.uint8)
            labels_tensor = torch.tensor(class_labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])        
        
        # Instances are clearly separated: iscrowd_tensor will be a tensor full of zeros
        iscrowd_tensor = torch.zeros((len(boxes_coord),), dtype=torch.int64)
        image_id_tensor = torch.tensor([idx])

        # Create the target dictionary for the R-CNN
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": image_id_tensor,
            "area": area_tensor,
            "iscrowd": iscrowd_tensor
        }

        # Cast image to tensor
        image_tensor = torch.as_tensor(image, dtype=torch.float32)

        return image_tensor, target
    
def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets for Mask R-CNN.
    This function will be passed to the DataLoader to ensure that batches are formed correctly.
    Args:
        batch: List of tuples (image_tensor, target_dict) returned by the dataset's __getitem__ method.
    Returns:
        A tuple of two lists: (list of image tensors, list of target dictionaries).
    """
    return tuple(zip(*batch))