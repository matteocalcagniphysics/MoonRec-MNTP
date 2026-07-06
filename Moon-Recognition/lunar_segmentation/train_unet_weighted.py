import os
import yaml
import torch
import torch.nn as nn
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import modules from the package
# We assume the script is run from Moon-Recognition/lunar_segmentation/
import sys
sys.path.append(os.path.abspath("."))

from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.datasets import MoonTileDataset
from lunar_segmentation.training.trainer import Trainer, BCEDiceLoss
from lunar_segmentation.data.preprocessing import CLASS_NAMES


class WeightedBCEDiceLoss(nn.Module):
    """BCEDice loss with per-class weights (class imbalance correction).

    Identical in structure to BCEDiceLoss, but the per-class weights are
    applied as pos_weight on the BCE term and as a per-class rescaling on
    the Dice term. Weights are registered as a buffer so they follow the
    model across devices and are saved in checkpoints. With uniform weights
    (all ones) this reduces to the original unweighted BCEDiceLoss.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer('w', class_weights)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.w.view(1, -1, 1, 1)
        )
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(0, 2, 3))
        den = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice = (1 - (num + 1e-6) / (den + 1e-6)) * self.w
        return bce * 0.5 + dice.mean() * 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmallUNet on lunar tiles")
    parser.add_argument(
        "--train-csv", type=str, required=True,
        help="Path to the CSV file for training (must have a 'tile_path' column)"
    )
    parser.add_argument(
        "--val-csv", type=str, required=True,
        help="Path to the CSV file for validation (must have a 'tile_path' column)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/unet_config.yaml",
        help="Path to the YAML config file (default: configs/unet_config.yaml)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory where the best model checkpoint will be saved"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of DataLoader workers (default: 2 — keep low when sharing CPU with GPU training)"
    )
    return parser.parse_args()


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV and validate that it contains the required 'tile_path' column."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")
    df = pd.read_csv(path)
    if 'tile_path' not in df.columns:
        raise ValueError(
            f"CSV '{path}' must contain a 'tile_path' column. "
            f"Found columns: {list(df.columns)}"
        )
    missing = df['tile_path'].apply(lambda p: not Path(p).exists())
    if missing.any():
        n = missing.sum()
        print(f"  [WARNING] {n}/{len(df)} tile paths in '{path.name}' do not exist on disk.")
    return df


def main():
    args = parse_args()

    # 1. Load configuration
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Training config: {config_path}")

    # 2. Load train and validation DataFrames from CSV
    print(f"\nTrain CSV: {args.train_csv}")
    train_df = load_csv(args.train_csv)
    print(f"  {len(train_df)} training tiles")

    print(f"Val CSV: {args.val_csv}")
    val_df = load_csv(args.val_csv)
    print(f"  {len(val_df)} validation tiles")

    # 3. Build Datasets and DataLoaders
    train_dataset = MoonTileDataset(train_df, augment=config.get('augment', True))
    val_dataset   = MoonTileDataset(val_df,   augment=False)  # no augmentation on val

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = SmallUNet(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_width=config.get('base_width', 32),
        depth=config.get('depth', 4),
        bottleneck_dropout=config.get('bottleneck_dropout', 0.3),
        decoder_dropout=config.get('decoder_dropout', 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['lr']),
        weight_decay=float(config.get('weight_decay', 0.01)),
    )

    # Optional: cosine annealing scheduler (from config)
    scheduler = None
    if config.get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', config['epochs']),
            eta_min=float(config.get('eta_min', 1e-6)),
        )

    # Weighted loss: read per-class weights from config['class_weights'].
    # Falls back to the original unweighted loss if the key is absent.
    if 'class_weights' in config:
        w = torch.tensor(
            [config['class_weights'][name] for name in CLASS_NAMES],
            dtype=torch.float32,
        ).to(device)
        criterion = WeightedBCEDiceLoss(w).to(device)
        print(f"Using WeightedBCEDiceLoss with per-class weights: "
              f"{dict(zip(CLASS_NAMES, [round(x, 1) for x in w.tolist()]))}")
    else:
        criterion = BCEDiceLoss()
        print("WARNING: 'class_weights' not found in config -- using unweighted BCEDiceLoss.")

    trainer = Trainer(model, optimizer, criterion, device=device)

    # 5. Training Loop
    epochs = config['epochs']
    patience = config.get('early_stopping_patience', epochs)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "unet_weighted_best.pth"

    print(f"\nStarting Training for {epochs} epochs (early stopping patience={patience})...\n")

    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)
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

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # LR step (after validation)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': best_val_loss,
                'train_csv': args.train_csv,
                'val_csv': args.val_csv,
            }, save_path)
            print(f"  saved to {save_path} (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nTraining Complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

