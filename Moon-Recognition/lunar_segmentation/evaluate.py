import os
import sys
import yaml
import torch
import argparse
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.abspath("."))

from lunar_segmentation.data.datasets import MoonTileDataset
from lunar_segmentation.data.preprocessing import CLASS_NAMES
from lunar_segmentation.evaluation.protocols import SemanticModelAdapter
from lunar_segmentation.evaluation.comparison import (
    evaluate_model, 
    compare_models, 
    significance_test,
    generate_report_table
)
from lunar_segmentation.visualization.eval_plotter import SegmentationVisualizer

def load_class(class_path: str):
    """Dynamically load a class from a full module path string."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Failed to load model class '{class_path}': {e}")

def load_model(model_config: dict, device: torch.device) -> object:
    """Instantiate a model and load its weights, supporting standard wrappers/modules."""
    model_name = model_config["name"]
    class_path = model_config["class"]
    checkpoint_path = Path(model_config["checkpoint_path"])
    args = model_config.get("args", {})
    
    print(f"\n--> Instantiating model: {model_name} ({class_path})")
    model_class = load_class(class_path)
    model = model_class(**args)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Weights file not found for {model_name} at: {checkpoint_path}")
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict wrapped in another dictionary or raw
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            
    # Clean 'module.' prefix if saved using DataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model

def run_dataset_threshold_sweep(
    model_adapter: object,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    from_logits: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Efficiently compute dataset-wide metrics for a sweep of threshold values."""
    num_classes = model_adapter.num_classes
    T = len(thresholds)
    
    global_tp = torch.zeros(T, num_classes, device=device)
    global_fp = torch.zeros(T, num_classes, device=device)
    global_fn = torch.zeros(T, num_classes, device=device)
    
    print(f"Sweeping {T} thresholds for {model_adapter.model_name}...")
    for images, masks in tqdm(dataloader, desc=f"Sweeping thresholds", leave=True):
        images = images.to(device)
        masks = masks.to(device).float()
        
        logits = model_adapter.predict(images)
        probs = torch.sigmoid(logits) if from_logits else logits
        
        for t_idx, t in enumerate(thresholds):
            preds = (probs > t).float()
            global_tp[t_idx] += (preds * masks).sum(dim=(0, 2, 3))
            global_fp[t_idx] += (preds * (1.0 - masks)).sum(dim=(0, 2, 3))
            global_fn[t_idx] += ((1.0 - preds) * masks).sum(dim=(0, 2, 3))
            
    iou_sweep = (global_tp + eps) / (global_tp + global_fp + global_fn + eps)
    dice_sweep = (2.0 * global_tp + eps) / (2.0 * global_tp + global_fp + global_fn + eps)
    
    return {
        "thresholds": torch.tensor(thresholds, dtype=torch.float32),
        "iou": iou_sweep.cpu(),
        "dice": dice_sweep.cpu(),
    }

def main():
    parser = argparse.ArgumentParser(description="Professional Model Evaluation Orchestrator")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml", help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None, help="Override data path")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override number of dataloader workers")
    parser.add_argument("--threshold", type=float, default=None, help="Override classification threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # 1. Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Resolve parameters
    data_root = Path(args.data_root or config["data"]["data_root"])
    # Resolve relative paths under workspace if not absolute
    if not data_root.is_absolute():
        script_dir = Path(__file__).resolve().parent
        potential_data_root = script_dir.parent / "data" / "MR" / "data" / "processed" / "tiles" / "marius_hills"
        if potential_data_root.exists():
            data_root = potential_data_root
        else:
            # check default Windows absolute path
            default_win_path = Path(r"C:\Users\Nicola Lavarda\Jupyter\LCP_moon\data\MR\data\processed\tiles\marius_hills")
            if default_win_path.exists():
                data_root = default_win_path

    batch_size = args.batch_size or config["data"].get("batch_size", 8)
    num_workers = args.num_workers if args.num_workers is not None else config["data"].get("num_workers", 0)
    threshold = args.threshold if args.threshold is not None else config["eval_params"].get("threshold", 0.5)
    from_logits = config["eval_params"].get("from_logits", True)
    
    # Output setup
    report_dir = Path(config["outputs"].get("report_dir", "outputs/evaluation"))
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print(" LUNAR SEGMENTATION EVALUATION ORCHESTRATOR ")
    print("="*60)
    print(f"Data root:   {data_root.resolve()}")
    print(f"Batch size:  {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Threshold:   {threshold}")
    print(f"Device:      {device}")
    print(f"Output dir:  {report_dir.resolve()}")
    print("="*60)

    # 3. Load dataset
    tile_paths = list(data_root.glob("*.npz"))
    if not tile_paths:
        print(f"Error: No .npz tiles found in: {data_root.resolve()}")
        return

    df = pd.DataFrame({'tile_path': [str(p) for p in tile_paths]})
    dataset = MoonTileDataset(df, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Loaded {len(df)} evaluation tiles.")

    # 4. Instantiate and evaluate models
    eval_results = {}
    model_configs = config.get("models", [])
    
    if not model_configs:
        print("Error: No models specified in configuration.")
        return

    for m_cfg in model_configs:
        model_name = m_cfg["name"]
        try:
            model = load_model(m_cfg, device)
            
            # Wrap in adapter if needed
            if hasattr(model, "predict") and hasattr(model, "model_name"):
                adapter = model
            else:
                adapter = SemanticModelAdapter(model=model, model_name=model_name)
                
            print(f"Evaluating {model_name}...")
            res = evaluate_model(
                model=adapter,
                dataloader=loader,
                class_names=CLASS_NAMES,
                from_logits=from_logits,
                threshold=threshold
            )
            eval_results[model_name] = res
            print(f"Finished {model_name}. Mean IoU: {res.mean_iou:.4f}")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not eval_results:
        print("Error: No models successfully evaluated.")
        return

    # 5. Statistical Significance and Comparison
    print("\n" + "="*50)
    print(" AGGREGATING RESULTS & GENERATING COMPARISONS ")
    print("="*50)
    
    # Bootstrap CI
    bootstrap_confidence = config["eval_params"].get("bootstrap_confidence", 0.95)
    n_resamples = config["eval_params"].get("n_resamples", 1000)
    
    comparison_df = compare_models(
        eval_results, 
        confidence=bootstrap_confidence, 
        n_resamples=n_resamples
    )
    
    report_table = generate_report_table(comparison_df, format="markdown")
    print("\n--- Summary Report Table ---")
    print(report_table)

    # 6. Save Report Document
    report_file_path = report_dir / "evaluation_report.md"
    with open(report_file_path, "w") as rf:
        rf.write(f"# Professional Evaluation Report: Marius Hills\n\n")
        rf.write(f"- **Evaluated Tiles**: {len(dataset)}\n")
        rf.write(f"- **Decision Threshold**: {threshold}\n")
        rf.write(f"- **Bootstrap Confidence**: {bootstrap_confidence} ({n_resamples} resamples)\n\n")
        
        rf.write("## Global & Per-Class Metrics\n\n")
        rf.write(report_table)
        rf.write("\n\n")
        
        # Pairwise Significance Table
        if len(eval_results) >= 2:
            rf.write("## Pairwise Statistical Significance (Wilcoxon Signed-Rank Test)\n\n")
            rf.write("Tests the null hypothesis that the metric distribution medians are equal. Significant differences ($p < 0.05$) are highlighted.\n\n")
            
            model_names = list(eval_results.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name_a, name_b = model_names[i], model_names[j]
                    rf.write(f"### {name_a} vs {name_b}\n\n")
                    try:
                        sig_df = significance_test(eval_results[name_a], eval_results[name_b], metric="iou")
                        rf.write(sig_df.to_markdown(index=False))
                        rf.write("\n\n")
                    except Exception as e:
                        rf.write(f"Failed to calculate Wilcoxon test: {e}\n\n")

    print(f"Report saved to: {report_file_path.resolve()}")

    # 7. Generate Visualizations
    plot_ops = config["outputs"].get("plots", {})
    if any(plot_ops.values()):
        print("\n" + "="*50)
        print(" GENERATING VISUALIZATION PLOTS ")
        print("="*50)
        
        viz = SegmentationVisualizer(class_names=CLASS_NAMES)
        
        # Metric Summary Bar Chart
        if plot_ops.get("metrics_summary", True):
            for model_name, res in eval_results.items():
                print(f"Generating metrics summary for {model_name}...")
                summary_df = res.summary_df()
                fig, _ = viz.plot_metrics_summary(summary_df, title=f"Metrics Summary - {model_name}")
                viz.save_figure(fig, plots_dir / f"metrics_summary_{model_name.lower().replace('-', '_')}.png")

        # Model Comparison Chart (if multiple models)
        if plot_ops.get("model_comparison", True) and len(eval_results) >= 2:
            print("Generating cross-model comparison plot...")
            fig, _ = viz.plot_model_comparison(comparison_df, metric="IoU", title="Model Comparison (IoU)")
            viz.save_figure(fig, plots_dir / "model_comparison_iou.png")
            fig, _ = viz.plot_model_comparison(comparison_df, metric="Dice", title="Model Comparison (Dice)")
            viz.save_figure(fig, plots_dir / "model_comparison_dice.png")

        # Select samples with features for qualitative plots
        max_samples = plot_ops.get("max_panel_samples", 5)
        interesting_indices = []
        for idx in range(len(dataset)):
            _, mask = dataset[idx]
            if mask.sum() > 50: # Find tiles with active features
                interesting_indices.append(idx)
                if len(interesting_indices) >= max_samples:
                    break
        if not interesting_indices:
            interesting_indices = list(range(min(len(dataset), max_samples)))
            
        # Qualitative panel plots for each model
        for model_name, res in eval_results.items():
            # Find the actual wrapper model or raw model
            for m_cfg in model_configs:
                if m_cfg["name"] == model_name:
                    model = load_model(m_cfg, device)
                    
            if plot_ops.get("prediction_panel", True):
                print(f"Generating prediction panels for {model_name}...")
                for idx_count, idx in enumerate(interesting_indices):
                    img_tensor, mask_tensor = dataset[idx]
                    
                    # Run prediction
                    with torch.no_grad():
                        pred_logits = model(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        probs = torch.sigmoid(pred_logits) if from_logits else pred_logits
                        pred_mask = (probs > threshold).float().numpy()
                        
                    img_np = img_tensor.numpy()
                    mask_np = mask_tensor.numpy()
                    
                    # For class 0 (e.g. impact_crater) or average
                    fig, _ = viz.plot_prediction_panel(
                        img_np, mask_np, pred_mask, class_idx=0,
                        title=f"{model_name} Prediction Panel - Sample {idx_count+1}"
                    )
                    viz.save_figure(fig, plots_dir / f"prediction_panel_{model_name.lower().replace('-', '_')}_sample_{idx_count+1}.png")

            if plot_ops.get("class_comparison", True):
                print(f"Generating class comparison grids for {model_name}...")
                for idx_count, idx in enumerate(interesting_indices):
                    img_tensor, mask_tensor = dataset[idx]
                    
                    with torch.no_grad():
                        pred_logits = model(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        probs = torch.sigmoid(pred_logits) if from_logits else pred_logits
                        pred_mask = (probs > threshold).float().numpy()
                        
                    img_np = img_tensor.numpy()
                    mask_np = mask_tensor.numpy()
                    
                    fig, _ = viz.plot_class_comparison(
                        img_np, mask_np, pred_mask,
                        title=f"{model_name} Per-Class Grid - Sample {idx_count+1}"
                    )
                    viz.save_figure(fig, plots_dir / f"class_comparison_{model_name.lower().replace('-', '_')}_sample_{idx_count+1}.png")

        # Threshold sensitivity sweep
        if plot_ops.get("threshold_sensitivity", True):
            for model_name in eval_results.keys():
                # Re-load adapter
                for m_cfg in model_configs:
                    if m_cfg["name"] == model_name:
                        model = load_model(m_cfg, device)
                        if hasattr(model, "predict") and hasattr(model, "model_name"):
                            adapter = model
                        else:
                            adapter = SemanticModelAdapter(model=model, model_name=model_name)
                            
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                sweep_res = run_dataset_threshold_sweep(
                    adapter, loader, device, thresholds, from_logits=from_logits
                )
                print(f"Generating threshold sensitivity plot for {model_name}...")
                fig, _ = viz.plot_threshold_sensitivity(sweep_res, title=f"Threshold Sensitivity - {model_name}")
                viz.save_figure(fig, plots_dir / f"threshold_sensitivity_{model_name.lower().replace('-', '_')}.png")

        print(f"\nAll plots saved to: {plots_dir.resolve()}")
        print("\nEvaluation Orchestrator Run Completed successfully!")

if __name__ == "__main__":
    main()
