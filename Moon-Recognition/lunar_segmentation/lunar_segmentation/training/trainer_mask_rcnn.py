import torch
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)

class MaskRCNN_Trainer:
    def __init__(self, model, optimizer, threshold=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        Args:
            model: PyTorch model
            optimizer: Optimizer istance
            threshold: Confidence threshold for predictions
            device: Device string (GPU/CPU, autodetected)
        """
        self.model = model
        self.optimizer = optimizer
        self.threshold = threshold
        self.device = device
        self.model.to(device)
        self.metric = MeanAveragePrecision().to(self.device)

    
    def train_one_epoch(self, loader):
        """
        Train model for one epoch.
        Args:
            loader: DataLoader for training data 
        Returns:
            avg_loss: Average loss for the epoch for logging
        """
        
        self.model.train()
        losses = []

        # Add a progress bar for training batches
        pbar = tqdm(loader, desc="Training Batch")

        for images, targets in pbar:
            self.optimizer.zero_grad()
            # Moving images and targets to the GPU
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # R-CNN gives a dict with 5 different losses
            loss_dict = self.model(images, targets)
            total_loss = torch.stack(list(loss_dict.values())).sum()   # Retrieve the total loss
            total_loss.backward()  # Backprop

            self.optimizer.step()

            # Save the loss value for logging and update the progress bar
            current_loss = total_loss.item()
            losses.append(current_loss)
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        return float(np.mean(losses)) if losses else np.nan


    def evaluate(self, loader):
        """
        Evaluate the model using Mean Average Precision (mAP) metric computed by torchmetrics.
        Args:
            loader: DataLoader for training data
        Returns:
            metrics_df: PandasDataFrame with mAP metrics for logging 
        """
        
        self.model.eval()

        with torch.no_grad():
            for images, targets in loader:
                # Moving images and targets to the GPU and getting predictions
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(images)  # Get predictions
                
                # Apply threshold to predictions using a mask to filter out low-confidence detections
                for output in outputs:
                    mask = output['scores'] > self.threshold
                    output['boxes'] = output['boxes'][mask]
                    output['labels'] = output['labels'][mask]
                    output['scores'] = output['scores'][mask]
                    output['masks'] = output['masks'][mask]

                # Metric update
                self.metric.update(outputs, targets)
            
            # Global mAP computation after processing all batches, then transform results to a DataFrame for logging
            mAP_dict = self.metric.compute()
            metrics_clean = {k: v.item() for k, v in mAP_dict.items()}
            metrics_df = pd.Series(metrics_clean).to_frame(name='value')
            
            logger.info("Evaluation Metrics:")
            logger.info(metrics_df)
            

            # Reset metric for the next evaluation phase
            self.metric.reset()
            
            return metrics_df
            