import os
import time
import datetime 
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Import custom modules for the Mask R-CNN
import sys
sys.path.insert(0, '/mnt/MoonRec-MNTP/Moon-Recognition/lunar_segmentation')
from lunar_segmentation.data.datasets import MoonTileTestDataset_RCNN, collate_fn
from lunar_segmentation.models.mask_rcnn import MaskRCNN
from lunar_segmentation.training.trainer_mask_rcnn import MaskRCNN_Trainer

# Exact Configuration from the Professor
BASE_DIR = Path('/mnt/MoonRec-MNTP/data/MR')
DATA_INDEX = BASE_DIR / 'tiles/index.csv'          # Training CSV file
MODEL_WEIGHTS_DIR = BASE_DIR / 'prova'           # Folder to save weights
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")

if DATA_INDEX.exists():
    print(f"Loading data from {DATA_INDEX}...")
    df = pd.read_csv(DATA_INDEX)
    
    # Put the right path to the tile in the dataframe
    df['tile_path'] = df['tile_path'].apply(lambda x: str(BASE_DIR / x))
    
    # Initialize Dataset and DataLoader (Using the RCNN Dataset and collate_fn!)
    dataset = MoonTileTestDataset_RCNN(index_df=df, augment=True)
    
    # IMPORTANT: Mask R-CNN requires collate_fn to handle multiple bounding boxes. 
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize Mask R-CNN Model
    print("Initializing Mask R-CNN model with pre-trained COCO weights...")
    model = MaskRCNN(num_classes=8, pretrained=True)
    model.to(DEVICE)
    
    # Optimizer and Trainer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    trainer = MaskRCNN_Trainer(model, optimizer, threshold=0.5, device=DEVICE)
    
    # Training Loop
    num_epochs = 5  # Set the desired number of epochs
    print(f"Starting training loop for {num_epochs} epochs...")
    
    # Use time.time() to track the start
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Run one training epoch
        loss = trainer.train_one_epoch(loader)
        print(f"Training epoch {epoch+1} completed. Average Loss: {loss:.4f}")
        
        # Save weights
        MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODEL_WEIGHTS_DIR / f'mask_rcnn_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Weights saved to {save_path}")

    print("\nTraining Completed!")
    
    # Correctly calculate the elapsed time
    total_time_seconds = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time_seconds)))
    
    print(f"Total training time: {total_time_str}")
    
    # Save the final model weights
    final_save_path = MODEL_WEIGHTS_DIR / 'mask_rcnn_final.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"Final weights saved to {final_save_path}")

else:
    print(f"Error: No training index found at {DATA_INDEX}.")