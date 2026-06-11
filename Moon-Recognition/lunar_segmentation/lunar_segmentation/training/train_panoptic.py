import sys
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# Import torchmetrics for the instance branch evaluation
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lunar_segmentation.data.datasets import MoonTileTestDataset_RCNN, panoptic_collate_fn
import lunar_segmentation.training.trainer as trainer
from lunar_segmentation.models.Panoptic_model.factory import build_models
from lunar_segmentation.models.Panoptic_model.fpn import PanopticFPN

BASEPATH = '/home/matteocalcagni/Desktop/MoonRec-MNTP/data/MR/'

# Load the dataset 
index_df = pd.read_csv(BASEPATH + 'tiles/index.csv')
index_df['tile_path'] = index_df['tile_path'].apply(lambda x: BASEPATH + x)

# Initialize Dataset 
dataset = MoonTileTestDataset_RCNN(index_df=index_df, augment=False, for_panoptic=True)

train_dataset, val_dataset = random_split(dataset, 
                                          [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=panoptic_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=panoptic_collate_fn)

# Define the building blocks of the panoptic architecture
backbone, semantic_branch, instance_branch = build_models(name='resnet18', num_classes=8, pretrained=True)

# Define the panoptic model
panoptic_model = PanopticFPN(backbone=backbone, semantic_branch=semantic_branch, instance_branch=instance_branch)

# Train the model
NEPOCHS = 1
optimizer = optim.Adam(panoptic_model.parameters(), lr=1e-4)

# Semantic Criterion 
criterion = trainer.PanopticBCEDiceLoss()

# Instance Metric
map_metric = MeanAveragePrecision()

# trainer
trainer_instance = trainer.PanopticTrainer(
    model=panoptic_model, 
    optimizer=optimizer, 
    criterion=criterion, 
    metric=map_metric,
    threshold=0.5, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


losses = []
metrics = []

for epoch in range(NEPOCHS):
    print(f"Starting Epoch {epoch + 1}/{NEPOCHS}...")
    
    # Train
    epoch_loss = trainer_instance.train_one_epoch(train_loader)
    losses.append(epoch_loss)
    print(f"Training Loss: {epoch_loss:.4f}")
    
    # Evaluate
    epoch_metrics = trainer_instance.evaluate(val_loader, criterion=criterion)
    metrics.append(epoch_metrics)

print("\nFinal Training Losses:", losses)