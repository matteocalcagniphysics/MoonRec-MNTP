"""
Data loader for lunar segmentation dataset with .npz files.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LunarSegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading lunar segmentation data from .npz files.
    
    Each .npz file contains:
    - 'image': (3, 256, 256) float32 normalized image
    - 'mask': (7, 256, 256) uint8 semantic segmentation masks
    - 'row': int64 row coordinate
    - 'col': int64 column coordinate
    """
    
    def __init__(self, 
                 data_dir: str,
                 file_pattern: str = "*.npz",
                 normalize: bool = True,
                 transform=None):
        """
        Args:
            data_dir (str): Path to directory containing .npz files
            file_pattern (str): Glob pattern to match .npz files. Default: "*.npz"
            normalize (bool): Whether image is already normalized. Default: True
            transform: Optional torchvision transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        
        # Find all matching .npz files
        self.file_paths = sorted(list(self.data_dir.glob(file_pattern)))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npz files found in {data_dir} matching {file_pattern}")
        
        print(f"Found {len(self.file_paths)} data files")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            image (torch.Tensor): Shape (3, 256, 256), float32
            mask (torch.Tensor): Shape (7, 256, 256), int64
            metadata (dict): Contains 'row' and 'col' coordinates
        """
        file_path = self.file_paths[idx]
        
        # Load .npz file
        data = np.load(file_path)
        
        # Extract image and mask
        image = data['image'].astype(np.float32)  # (3, 256, 256)
        mask = data['mask'].astype(np.int64)       # (7, 256, 256)
        
        # Extract metadata
        row = int(data['row'])
        col = int(data['col'])
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        metadata = {
            'row': row,
            'col': col,
            'filename': file_path.name
        }
        
        return image_tensor, mask_tensor, metadata


def create_dataloader(data_dir: str,
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     file_pattern: str = "*.npz") -> DataLoader:
    """
    Create a PyTorch DataLoader for the lunar segmentation dataset.
    
    Args:
        data_dir (str): Path to directory containing .npz files
        batch_size (int): Batch size. Default: 4
        shuffle (bool): Whether to shuffle data. Default: True
        num_workers (int): Number of workers for data loading. Default: 0
        file_pattern (str): Glob pattern for .npz files. Default: "*.npz"
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = LunarSegmentationDataset(
        data_dir=data_dir,
        file_pattern=file_pattern,
        normalize=True
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable-sized batches"""
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]
        
        return images, masks, metadata
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


if __name__ == '__main__':
    # Example usage
    data_dir = '../data'
    
    # Create dataset
    dataset = LunarSegmentationDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    image, mask, metadata = dataset[0]
    print(f"\nSingle sample:")
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Metadata: {metadata}")
    
    # Create dataloader
    dataloader = create_dataloader(data_dir, batch_size=2)
    print(f"\nDataLoader created with batch size: 2")
    
    # Test batch
    images, masks, metadata_list = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Batch metadata: {metadata_list}")
