import sys
import yaml
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import torchmetrics for the instance branch evaluation
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lunar_segmentation.data.datasets import MoonTileTestDataset_RCNN, panoptic_collate_fn
import lunar_segmentation.training.trainer as trainer_module
from lunar_segmentation.models.PAN4_factory import build_models
from lunar_segmentation.models.PAN3_fpn import PanopticFPN

# ── Load config ────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT_DIR / 'configs' / 'panoptic_config.yaml'
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)
# Find repo root dynamically
repo_root = ROOT_DIR
for parent in [ROOT_DIR] + list(ROOT_DIR.parents):
    if (parent / ".git").exists() or (parent / "Moon-Recognition").exists():
        repo_root = parent
        break

BASEPATH = str((repo_root / 'data' / 'MR').resolve()) + '/'
MODEL_WEIGHTS_DIR = Path(BASEPATH) / 'panoptic_weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = cfg.get('batch_size', 4)
LR         = cfg.get('lr', 1e-4)
NEPOCHS    = cfg.get('epochs', 20)

# Build the class-weights dict from the YAML (may be None if key is absent)
CLASS_WEIGHTS = cfg.get('class_weights', None)   # dict or None

print(f"Device: {DEVICE}")
print(f"Config loaded from {CONFIG_PATH}")
if CLASS_WEIGHTS:
    print(f"Class weights: {CLASS_WEIGHTS}")

# ── Load the dataset ────────────────────────────────────────────────────────
train_index_df = pd.read_csv(ROOT_DIR / "train_index.csv")
val_index_df   = pd.read_csv(ROOT_DIR / "val_index.csv")

train_index_df['tile_path'] = train_index_df['tile_path'].apply(lambda x: BASEPATH + x)
val_index_df['tile_path']   = val_index_df['tile_path'].apply(lambda x: BASEPATH + x)

train_dataset = MoonTileTestDataset_RCNN(index_df=train_index_df, augment=False, for_panoptic=True)
val_dataset   = MoonTileTestDataset_RCNN(index_df=val_index_df,   augment=False, for_panoptic=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=panoptic_collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=panoptic_collate_fn)

# ── Build model ─────────────────────────────────────────────────────────────
backbone, semantic_branch, instance_branch = build_models(name='resnet18', num_classes=8, pretrained=True)
panoptic_model = PanopticFPN(backbone=backbone, semantic_branch=semantic_branch, instance_branch=instance_branch)

# ── Optimizer & loss ─────────────────────────────────────────────────────────
optimizer = optim.Adam(panoptic_model.parameters(), lr=LR)

# Semantic criterion — weights are injected by PanopticTrainer after converting
# the dict to a tensor, so we initialise without them here.
criterion = trainer_module.PanopticBCEDiceLoss()

# Instance metric
map_metric = MeanAveragePrecision()

# ── Trainer (passes class_weights to both branches) ──────────────────────────
trainer_instance = trainer_module.PanopticTrainer(
    model=panoptic_model,
    optimizer=optimizer,
    criterion=criterion,
    metric=map_metric,
    threshold=0.5,
    device=DEVICE,
    class_weights=CLASS_WEIGHTS,   # dict loaded from YAML
)

# ── Training loop ─────────────────────────────────────────────────────────────
losses  = []
metrics = []

for epoch in range(NEPOCHS):
    print(f"Starting Epoch {epoch + 1}/{NEPOCHS}...")

    epoch_loss = trainer_instance.train_one_epoch(train_loader)
    losses.append(epoch_loss)
    print(f"Training Loss: {epoch_loss:.4f}")

    epoch_metrics = trainer_instance.evaluate(val_loader, criterion=criterion)
    metrics.append(epoch_metrics)

    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_WEIGHTS_DIR / f'panoptic_epoch_{epoch + 1}.pth'
    torch.save(panoptic_model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

# ── Save final weights ────────────────────────────────────────────────────────
final_save_path = MODEL_WEIGHTS_DIR / 'panoptic_final.pth'
torch.save(panoptic_model.state_dict(), final_save_path)
print(f"Final weights saved to {final_save_path}")

print("\nFinal Training Losses:", losses)
