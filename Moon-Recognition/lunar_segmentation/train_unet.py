import os
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# Import modules from the package
# We assume the script is run from Moon-Recognition/lunar_segmentation/
import sys
sys.path.append(os.path.abspath("."))

from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.datasets import MoonTileDataset
from lunar_segmentation.training.trainer import Trainer, BCEDiceLoss

def main():
    # 1. Load configuration
    config_path = Path("configs/unet_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Training Configuration Loaded from {config_path} ---")
    
    # 2. Prepare Data Index
    # We look for all .npz files in the data directory provided by the user
    # Or fallback to the local processed directory
    data_root = Path(r"C:\Users\Nicola Lavarda\Jupyter\LCP_moon\data\MR\data\processed\tiles\marius_hills")
    if not data_root.exists():
        data_root = Path("data/processed/tiles/marius_hills")
    
    print(f"Scanning tiles in: {data_root}")
    tile_paths = list(data_root.glob("*.npz"))
    if not tile_paths:
        print("Error: No .npz tiles found!")
        return

    df = pd.DataFrame({'tile_path': [str(p) for p in tile_paths]})
    print(f"Found {len(df)} tiles.")

    # 3. Split Dataset (90% train, 10% val)
    full_dataset = MoonTileDataset(df, augment=config.get('augment', True))
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False 

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 4. Initialize Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SmallUNet(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_width=config.get('base_width', 32),
        depth=config.get('depth', 4),
        bottleneck_dropout=config.get('bottleneck_dropout', 0.3),
        decoder_dropout=config.get('decoder_dropout', 0.1)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['lr']), 
        weight_decay=float(config.get('weight_decay', 0.01))
    )
    
    criterion = BCEDiceLoss()
    trainer = Trainer(model, optimizer, criterion, device=device)

    # 5. Training Loop
    epochs = config['epochs']
    best_val_loss = float('inf')
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "unet_best.pth"

    print(f"\nStarting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase with tqdm
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': best_val_loss
            }, save_path)
            print(f"--> Saved best model to {save_path}")

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
