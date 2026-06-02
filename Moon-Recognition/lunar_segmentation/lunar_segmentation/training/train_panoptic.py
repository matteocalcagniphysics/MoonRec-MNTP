import sys
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lunar_segmentation.data.datasets import MoonTileDataset
import lunar_segmentation.training.trainer as trainer
from lunar_segmentation.models.Panoptic_model.factory import make_fpn_resnet

BASEPATH = '/home/matteocalcagni/Desktop/MoonRec-MNTP/data/MR/'

# Load the dataset 
index_df = pd.read_csv(BASEPATH + 'tiles/index.csv')
index_df['tile_path'] = index_df['tile_path'].apply(lambda x: BASEPATH + x)
dataset = MoonTileDataset(index_df=index_df, augment=False)

train_dataset, val_dataset = random_split(dataset, 
                                          [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define the current model
panoptic_model = make_fpn_resnet(num_classes=7, fpn_type='panoptic', pretrained=True)

# Train the model
NEPOCHS = 1

optimizer = optim.Adam(panoptic_model.parameters(), lr=1e-4)
criterion = trainer.BCEDiceLoss()
trainer_instance = trainer.Trainer(model=panoptic_model, optimizer=optimizer, criterion=criterion)

for epoch in range(NEPOCHS):
    losses = []
    metrics = []
    losses.append(trainer_instance.train_one_epoch(train_loader))
    metrics.append(trainer_instance.evaluate(val_loader, criterion=criterion))

print(losses)
print(metrics)