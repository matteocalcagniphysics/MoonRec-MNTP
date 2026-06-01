"""
Example script: Using the Panoptic FPN model with lunar segmentation data.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from factory import make_fpn_resnet
from data_loader import create_dataloader


def main():
    """Demonstrate model inference on lunar segmentation data."""
    
    print("=" * 70)
    print("Lunar Segmentation with Panoptic FPN")
    print("=" * 70)
    
    # ============================================================================
    # 1. CREATE MODEL
    # ============================================================================
    print("\n[1] Creating model...")
    model = make_fpn_resnet(
        name='resnet18',           # ResNet-18 backbone
        fpn_type='panoptic',       # Panoptic FPN architecture
        out_size=(256, 256),       # Output size matches data: 256x256
        fpn_channels=256,          # Hidden channels in FPN
        num_classes=7,             # Number of semantic classes (from data)
        pretrained=True,           # Use ImageNet pretrained weights
        in_channels=3              # RGB input
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model created on device: {device}")
    print(f"  - Architecture: ResNet-18 + Panoptic FPN")
    print(f"  - Input: (B, 3, 256, 256)")
    print(f"  - Output: (B, 7, 256, 256)")
    
    # ============================================================================
    # 2. CREATE DATA LOADER
    # ============================================================================
    print("\n[2] Creating data loader...")
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print("  Please ensure data files are in the 'data/' directory")
        return
    
    dataloader = create_dataloader(
        data_dir=str(data_dir),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        file_pattern='marius_hills_r*.npz'
    )
    
    print(f"✓ Data loader created")
    print(f"  - Data directory: {data_dir}")
    print(f"  - Batch size: 4")
    print(f"  - Total batches: {len(dataloader)}")
    
    # ============================================================================
    # 3. RUN INFERENCE ON FIRST BATCH
    # ============================================================================
    print("\n[3] Running inference on first batch...")
    
    with torch.no_grad():
        # Get first batch
        images, masks, metadata = next(iter(dataloader))
        images = images.to(device)
        
        print(f"  - Input images shape: {images.shape}")
        print(f"  - Ground truth masks shape: {masks.shape}")
        print(f"  - Metadata: {metadata}")
        
        # Forward pass
        predictions = model(images)
        
        print(f"\n✓ Inference successful!")
        print(f"  - Output predictions shape: {predictions.shape}")
        print(f"  - Output dtype: {predictions.dtype}")
        print(f"  - Output device: {predictions.device}")
        
        # Get class predictions
        class_predictions = torch.argmax(predictions, dim=1)  # (B, 256, 256)
        print(f"  - Class predictions shape: {class_predictions.shape}")
        print(f"  - Unique predicted classes: {torch.unique(class_predictions).tolist()}")
    
    # ============================================================================
    # 4. DETAILED STATISTICS
    # ============================================================================
    print("\n[4] Detailed statistics...")
    
    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Data statistics
    print(f"\n  - Data statistics (from batch):")
    print(f"    - Image range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"    - Image mean: {images.mean():.4f}, std: {images.std():.4f}")
    print(f"    - Mask classes: {torch.unique(masks).tolist()}")
    
    # Output statistics
    print(f"\n  - Model output statistics:")
    print(f"    - Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    - Predictions mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
    
    # ============================================================================
    # 5. SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Model can successfully process lunar segmentation data!")
    print(f"  - Input: RGB images from .npz files (256×256)")
    print(f"  - Output: Semantic segmentation masks (7 classes, 256×256)")
    print(f"  - Pipeline: Data Loader → Model → Predictions")
    print("\nNext steps:")
    print("  1. Train the model on the full dataset")
    print("  2. Evaluate predictions against ground truth masks")
    print("  3. Fine-tune hyperparameters for better performance")
    print("=" * 70)


if __name__ == '__main__':
    main()
