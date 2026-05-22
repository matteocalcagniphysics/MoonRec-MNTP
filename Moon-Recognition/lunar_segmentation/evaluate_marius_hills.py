import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Import modules from the package
import sys
sys.path.append(os.path.abspath("."))

from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.datasets import MoonTileDataset
from lunar_segmentation.data.preprocessing import CLASS_NAMES
from lunar_segmentation.evaluation.protocols import SemanticModelAdapter
from lunar_segmentation.evaluation.comparison import evaluate_model, generate_report_table

def main():
    # 1. Load configuration
    config_path = Path("configs/unet_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Evaluation Configuration Loaded from {config_path} ---")

    # 2. Initialize Model and Load Weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = Path("outputs/unet_best.pth")
    
    if not weights_path.exists():
        print(f"Error: Trained weights not found at {weights_path}. Please run train_unet.py first.")
        return

    model = SmallUNet(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        base_width=config.get('base_width', 32),
        depth=config.get('depth', 4),
        bottleneck_dropout=config.get('bottleneck_dropout', 0.3),
        decoder_dropout=config.get('decoder_dropout', 0.1)
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded weights from {weights_path} (Best Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")

    # 3. Prepare Evaluation Data
    # Resolve path relative to this script's directory, or fall back to specified paths
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent.parent / "data" / "MR" / "data" / "processed" / "tiles" / "marius_hills"
    
    if not data_root.exists():
        data_root = Path(r"C:\Users\Nicola Lavarda\Jupyter\LCP_moon\data\MR\data\processed\tiles\marius_hills")
    if not data_root.exists():
        data_root = Path("data/processed/tiles/marius_hills")
    
    print(f"Scanning evaluation tiles in: {data_root.resolve()}")
    tile_paths = list(data_root.glob("*.npz"))
    if not tile_paths:
        print(f"Error: No .npz tiles found in: {data_root.resolve()}")
        return

    df = pd.DataFrame({'tile_path': [str(p) for p in tile_paths]})
    dataset = MoonTileDataset(df, augment=False)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    print(f"Found {len(df)} tiles for evaluation.")

    # 4. Run Evaluation
    print("\nRunning evaluation...")
    adapter = SemanticModelAdapter(model=model, model_name="SmallUNet-Best")
    
    # Run orchestration
    results = evaluate_model(
        model=adapter, 
        dataloader=loader, 
        class_names=CLASS_NAMES,
        threshold=config.get('adjusted_threshold', 0.5)
    )

    # 5. Show Report
    print("\n" + "="*50)
    print(" EVALUATION REPORT: MARIUS HILLS ")
    print("="*50)
    
    # Summary Table
    summary_df = results.summary_df()
    markdown_table = generate_report_table(pd.concat([summary_df]))
    print(markdown_table)
    
    print(f"\nOverall Mean IoU: {results.mean_iou:.4f}")
    
    # Save report
    report_path = Path("outputs/evaluation_report_marius_hills.md")
    with open(report_path, "w") as f:
        f.write(f"# Evaluation Report: Marius Hills\n\n")
        f.write(f"- **Model**: {adapter.model_name}\n")
        f.write(f"- **Weights**: {weights_path}\n")
        f.write(f"- **Mean IoU**: {results.mean_iou:.4f}\n\n")
        f.write("## Metrics per Class\n\n")
        f.write(markdown_table)
    
    print(f"\nFull report saved to {report_path}")

if __name__ == "__main__":
    main()
