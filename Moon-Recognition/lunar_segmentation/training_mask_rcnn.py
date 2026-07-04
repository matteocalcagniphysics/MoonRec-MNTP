import time
import datetime
import logging
import sys
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Import custom modules for the Mask R-CNN
sys.path.insert(0, '/LCP/MoonRec-MNTP/Moon-Recognition/lunar_segmentation')
from lunar_segmentation.data.datasets import MoonTileTestDataset_RCNN, collate_fn
from lunar_segmentation.models.mask_rcnn import MaskRCNN
from lunar_segmentation.training.trainer_mask_rcnn import MaskRCNN_Trainer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('training_mask_rcnn')

# Configuration of the Paths and Hyperparameters
BASE_DIR          = Path('/LCP/MoonRec-MNTP/data/MR')
TRAIN_CSV         = BASE_DIR / 'tiles/train.csv'   # produced by run_preprocess.py
VAL_CSV           = BASE_DIR / 'tiles/val.csv'     # produced by run_preprocess.py
MODEL_WEIGHTS_DIR = BASE_DIR / 'weights'           # folder to save checkpoints

BATCH_SIZE  = 8
NUM_EPOCHS  = 3
LR          = 1e-4
NUM_WORKERS = 15
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

logger.info(f"Device: {DEVICE}")

# Helper: load a split CSV and fix tile paths
def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['tile_path'] = df['tile_path'].apply(lambda x: str(BASE_DIR / x))
    return df


# Check if the train/val csv files exist
if not TRAIN_CSV.exists() or not VAL_CSV.exists():
    logger.error(
        f"Split CSVs not found. Expected:\n  {TRAIN_CSV}\n  {VAL_CSV}\n"
        "Run run_preprocess.py first."
    )
    sys.exit(1)


# Load the train/val split CSV files
logger.info(f"Loading train split from {TRAIN_CSV} ...")
train_df = load_split(TRAIN_CSV)
logger.info(f"Loading val split from {VAL_CSV} ...")
val_df   = load_split(VAL_CSV)

logger.info(f"Train tiles: {len(train_df)} | Val tiles: {len(val_df)}")

# Prepare the training and validation datasets
train_dataset = MoonTileTestDataset_RCNN(index_df=train_df, augment=True)
val_dataset   = MoonTileTestDataset_RCNN(index_df=val_df,   augment=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)


# Initialize the Mask R-CNN model with pre-trained COCO weights
logger.info("Initialising Mask R-CNN with pre-trained COCO weights ...")
model = MaskRCNN(num_classes=8, pretrained=True)
model.to(DEVICE)

params    = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LR)
trainer   = MaskRCNN_Trainer(model, optimizer, threshold=0.5, device=DEVICE)

MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Training Loop
logger.info(f"Starting training for {NUM_EPOCHS} epochs ...")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    logger.info(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")

    # Train
    train_loss = trainer.train_one_epoch(train_loader)
    logger.info(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate
    metrics_df = trainer.evaluate(val_loader)
    map_val = metrics_df.loc['map', 'value'] if 'map' in metrics_df.index else float('nan')
    logger.info(f"[Epoch {epoch}] Val mAP: {map_val:.4f}")

    # Save checkpoint
    ckpt_path = MODEL_WEIGHTS_DIR / f'mask_rcnn_epoch_{epoch:02d}.pth'
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

# Save the final trained model weights
total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
logger.info(f"Training completed in {total_time}.")

final_path = MODEL_WEIGHTS_DIR / 'mask_rcnn_final.pth'
torch.save(model.state_dict(), final_path)
logger.info(f"Final weights saved: {final_path}")