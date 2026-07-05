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
from lunar_segmentation.evaluation.protocols import create_adapter
import lunar_segmentation.evaluation.mask_rcnn_adapter  # noqa: F401  (register "instance" adapter)
import lunar_segmentation.evaluation.panoptic_adapter  # noqa: F401  (register "panoptic" adapter)
from lunar_segmentation.evaluation.comparison import (
    EvaluationResult,
    evaluate_model,
    compare_models,
    significance_test,
    generate_report_table,
)
from lunar_segmentation.visualization.eval_plotter import SegmentationVisualizer


# ======================================================================== #
#  Dynamic class loading                                                    #
# ======================================================================== #

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
    model_name      = model_config["name"]
    class_path      = model_config["class"]
    checkpoint_path = Path(model_config["checkpoint_path"])
    args            = model_config.get("args", {})

    print(f"\n--> Instantiating model: {model_name} ({class_path})")
    model_class = load_class(class_path)
    model       = model_class(**args)

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

    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model


# ======================================================================== #
#  EvaluationResult  ←→  disk  (npz)                                       #
# ======================================================================== #

def save_eval_result(result: EvaluationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model_name=result.model_name,
        class_names=result.class_names,
        per_sample_iou=result.per_sample_iou,
        per_sample_dice=result.per_sample_dice,
        per_sample_precision=result.per_sample_precision,
        per_sample_recall=result.per_sample_recall,
        per_sample_f1=result.per_sample_f1,
    )
    print(f"  [cache] EvaluationResult saved -> {path}")


def load_eval_result(path: Path) -> EvaluationResult:
    """Load an EvaluationResult previously saved with save_eval_result."""
    data = np.load(path, allow_pickle=True)
    return EvaluationResult(
        model_name=str(data["model_name"]),
        class_names=list(data["class_names"]),
        per_sample_iou=data["per_sample_iou"],
        per_sample_dice=data["per_sample_dice"],
        per_sample_precision=data["per_sample_precision"],
        per_sample_recall=data["per_sample_recall"],
        per_sample_f1=data["per_sample_f1"],
    )


# ======================================================================== #
#  Threshold sweep cache  ←→  disk  (aggregated TP/FP/FN per threshold)    #
# ======================================================================== #

_SWEEP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _sweep_cache_path(cache_root: Path, model_name: str) -> Path:
    return cache_root / f"sweep_{model_name.lower().replace('-', '_').replace(' ', '_')}.pt"


def save_sweep_accumulators(
    global_tp: torch.Tensor,
    global_fp: torch.Tensor,
    global_fn: torch.Tensor,
    thresholds: list[float],
    path: Path,
) -> None:
    """Save accumulated TP/FP/FN tensors and threshold list to disk."""
    torch.save(
        {
            "global_tp": global_tp,
            "global_fp": global_fp,
            "global_fn": global_fn,
            "thresholds": thresholds,
        },
        path,
    )
    print(f"  [cache] Sweep accumulators saved -> {path}")


def load_sweep_accumulators(path: Path):
    """Load previously saved sweep accumulators."""
    data = torch.load(path, weights_only=True)
    return data["global_tp"], data["global_fp"], data["global_fn"], data["thresholds"]


def compute_sweep_result(
    global_tp: torch.Tensor,
    global_fp: torch.Tensor,
    global_fn: torch.Tensor,
    thresholds: list[float],
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    iou_sweep  = (global_tp + eps) / (global_tp + global_fp + global_fn + eps)
    dice_sweep = (2.0 * global_tp + eps) / (2.0 * global_tp + global_fp + global_fn + eps)
    return {
        "thresholds": torch.tensor(thresholds, dtype=torch.float32),
        "iou":  iou_sweep.cpu(),
        "dice": dice_sweep.cpu(),
    }


# ======================================================================== #
#  Combined evaluate + sweep accumulation (single forward pass)            #
# ======================================================================== #

def _evaluate_and_build_sweep(
    adapter,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
    from_logits: bool,
    threshold: float,
    sweep_thresholds: list[float],
) -> tuple["EvaluationResult", torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass that simultaneously computes per-sample metrics
    AND accumulates global TP/FP/FN for every threshold in *sweep_thresholds*.

    Returns
    -------
    (EvaluationResult, global_tp, global_fp, global_fn)
        The TP/FP/FN tensors have shape (T, C) and live on CPU.
    """
    from lunar_segmentation.evaluation.metrics import compute_all_metrics_vectorized

    T = len(sweep_thresholds)
    global_tp = None
    global_fp = None
    global_fn = None

    all_iou:  list[np.ndarray] = []
    all_dice: list[np.ndarray] = []
    all_prec: list[np.ndarray] = []
    all_rec:  list[np.ndarray] = []
    all_f1:   list[np.ndarray] = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Evaluating {adapter.model_name}", leave=True):
            images    = images.to(device)
            masks_dev = masks.to(device).float()

            logits = adapter.predict(images)
            probs  = torch.sigmoid(logits) if from_logits else logits
            del logits, images

            # ── Per-sample metrics (for EvaluationResult) ──────────────────
            m = compute_all_metrics_vectorized(
                probs, masks_dev,
                from_logits=False,
                threshold=threshold,
            )
            all_iou.append(m["iou"].cpu().numpy())
            all_dice.append(m["dice"].cpu().numpy())
            all_prec.append(m["precision"].cpu().numpy())
            all_rec.append(m["recall"].cpu().numpy())
            all_f1.append(m["f1"].cpu().numpy())

            # ── Global sweep accumulators (TP/FP/FN per threshold) ─────────
            # Vectorized: compare all thresholds in one shot, staying on device.
            # thresh_tensor shape: (T, 1, 1, 1, 1) → broadcast over (B, C, H, W)
            thresh_tensor = torch.tensor(
                sweep_thresholds, dtype=probs.dtype, device=probs.device
            ).view(T, 1, 1, 1, 1)
            preds_all = (probs.unsqueeze(0) > thresh_tensor).float()  # (T, B, C, H, W)

            if global_tp is None:
                num_classes = probs.shape[1]
                global_tp = torch.zeros(T, num_classes, device=probs.device)
                global_fp = torch.zeros(T, num_classes, device=probs.device)
                global_fn = torch.zeros(T, num_classes, device=probs.device)

            global_tp += (preds_all *           masks_dev.unsqueeze(0)).sum(dim=(1, 3, 4))
            global_fp += (preds_all * (1.0 -    masks_dev.unsqueeze(0))).sum(dim=(1, 3, 4))
            global_fn += ((1.0 - preds_all) *   masks_dev.unsqueeze(0)).sum(dim=(1, 3, 4))
            del preds_all, probs, masks_dev

    result = EvaluationResult(
        model_name=adapter.model_name,
        class_names=class_names,
        per_sample_iou=np.concatenate(all_iou,  axis=0) if all_iou  else np.array([]),
        per_sample_dice=np.concatenate(all_dice, axis=0) if all_dice else np.array([]),
        per_sample_precision=np.concatenate(all_prec, axis=0) if all_prec else np.array([]),
        per_sample_recall=np.concatenate(all_rec,  axis=0) if all_rec  else np.array([]),
        per_sample_f1=np.concatenate(all_f1,   axis=0) if all_f1   else np.array([]),
    )
    return result, global_tp, global_fp, global_fn


# ======================================================================== #
#  Main                                                                     #
# ======================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Professional Model Evaluation Orchestrator")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.yaml",
                        help="Path to config file")
    parser.add_argument("--data_root",   type=str,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--num_workers", type=int,   default=None)
    parser.add_argument("--threshold",   type=float, default=None)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing caches and rerun from scratch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit evaluation to a random subset of N tiles")
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
    if not data_root.is_absolute():
        script_dir = Path(__file__).resolve().parent
        potential  = script_dir.parent.parent / "data" / "MR" / "data" / "processed" / "tiles" / "marius_hills"
        if potential.exists():
            data_root = potential

    batch_size  = args.batch_size or config["data"].get("batch_size", 8)
    num_workers = args.num_workers if args.num_workers is not None else config["data"].get("num_workers", 0)
    threshold   = args.threshold   if args.threshold   is not None else config["eval_params"].get("threshold", 0.5)
    from_logits = config["eval_params"].get("from_logits", True)

    report_dir = Path(config["outputs"].get("report_dir", "outputs/evaluation"))
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir  = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    cache_dir  = report_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

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
    print(f"Cache dir:   {cache_dir.resolve()}")
    print(f"Force rerun: {args.force}")
    if args.limit is not None:
        print(f"Limit:       {args.limit} random tiles")
    print("="*60)

    # 3. Load dataset
    tile_paths = list(data_root.glob("*.npz"))
    if not tile_paths:
        print(f"Error: No .npz tiles found in: {data_root.resolve()}")
        return

    df      = pd.DataFrame({"tile_path": [str(p) for p in tile_paths]})
    if args.limit is not None:
        limit_n = min(args.limit, len(df))
        df = df.sample(n=limit_n, random_state=42).reset_index(drop=True)
        print(f"Randomly selected {limit_n} tiles out of {len(tile_paths)} for quick evaluation.")

    dataset = MoonTileDataset(df, augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Loaded {len(df)} evaluation tiles ({len(loader)} batches).")

    # 4. Evaluate models (with cache)
    eval_results:  dict[str, EvaluationResult]  = {}
    sweep_results: dict[str, dict]              = {}
    adapter_cache: dict[str, object]            = {}
    model_configs  = config.get("models", [])
    plot_ops       = config["outputs"].get("plots", {})

    if not model_configs:
        print("Error: No models specified in configuration.")
        return

    for m_cfg in model_configs:
        model_name   = m_cfg["name"]
        result_cache = cache_dir / f"eval_result_{model_name.lower().replace('-','_').replace(' ','_')}.npz"
        sweep_cache  = _sweep_cache_path(cache_dir, model_name)

        eval_cached  = (not args.force) and result_cache.exists()
        sweep_cached = (not args.force) and sweep_cache.exists()

        if eval_cached and sweep_cached:
            # ── Both already on disk — load and skip inference entirely ──
            print(f"\n[cache] Loading EvaluationResult + sweep for {model_name}...")
            res = load_eval_result(result_cache)
            eval_results[model_name] = res
            print(f"  Mean IoU: {res.mean_iou:.4f}  (loaded from cache)")

            tp, fp, fn, thr = load_sweep_accumulators(sweep_cache)
            sweep_results[model_name] = compute_sweep_result(tp, fp, fn, thr)
            print(f"  Sweep accumulators loaded from cache.")

        else:
            # ── Need to run inference ──────────────────────────────────
            try:
                model        = load_model(m_cfg, device)
                model_type   = m_cfg.get("type", "semantic")
                adapter_args = m_cfg.get("adapter_args", {})
                adapter      = create_adapter(
                    model=model, model_name=model_name,
                    model_type=model_type, **adapter_args,
                )
                adapter_cache[model_name] = adapter

                # Respect the adapter's output format
                effective_from_logits = adapter.output_is_logits

                need_sweep = plot_ops.get("threshold_sensitivity", True) and not sweep_cached
                need_eval  = not eval_cached

                if need_eval and need_sweep:
                    print(f"\nEvaluating {model_name} (metrics + sweep in one pass)...")
                    res, g_tp, g_fp, g_fn = _evaluate_and_build_sweep(
                        adapter, loader, device, CLASS_NAMES,
                        from_logits=effective_from_logits,
                        threshold=threshold,
                        sweep_thresholds=_SWEEP_THRESHOLDS,
                    )
                    save_eval_result(res, result_cache)
                    save_sweep_accumulators(g_tp, g_fp, g_fn, _SWEEP_THRESHOLDS, sweep_cache)
                    sweep_results[model_name] = compute_sweep_result(g_tp, g_fp, g_fn, _SWEEP_THRESHOLDS)

                elif need_eval and not need_sweep:
                    print(f"\nEvaluating {model_name} (metrics only)...")
                    res = evaluate_model(
                        model=adapter, dataloader=loader, class_names=CLASS_NAMES,
                        from_logits=effective_from_logits, threshold=threshold,
                    )
                    save_eval_result(res, result_cache)

                elif not need_eval and need_sweep:
                    print(f"\n[cache] EvaluationResult for {model_name} already cached.")
                    res = load_eval_result(result_cache)
                    print(f"  Running sweep-only pass...")
                    _, g_tp, g_fp, g_fn = _evaluate_and_build_sweep(
                        adapter, loader, device, CLASS_NAMES,
                        from_logits=effective_from_logits,
                        threshold=threshold,
                        sweep_thresholds=_SWEEP_THRESHOLDS,
                    )
                    save_sweep_accumulators(g_tp, g_fp, g_fn, _SWEEP_THRESHOLDS, sweep_cache)
                    sweep_results[model_name] = compute_sweep_result(g_tp, g_fp, g_fn, _SWEEP_THRESHOLDS)

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

    bootstrap_confidence = config["eval_params"].get("bootstrap_confidence", 0.95)
    n_resamples          = config["eval_params"].get("n_resamples", 1000)

    comparison_df = compare_models(eval_results, confidence=bootstrap_confidence, n_resamples=n_resamples)
    report_table  = generate_report_table(comparison_df, format="markdown")
    print("\n--- Summary Report Table ---")
    print(report_table)

    # 6. Save Report Document
    report_file_path = report_dir / "evaluation_report.md"
    with open(report_file_path, "w") as rf:
        rf.write("# Professional Evaluation Report: Marius Hills\n\n")
        rf.write(f"- **Evaluated Tiles**: {len(dataset)}\n")
        rf.write(f"- **Decision Threshold**: {threshold}\n")
        rf.write(f"- **Bootstrap Confidence**: {bootstrap_confidence} ({n_resamples} resamples)\n\n")

        rf.write("## Global & Per-Class Metrics\n\n")
        rf.write(report_table)
        rf.write("\n\n")

        if len(eval_results) >= 2:
            rf.write("## Pairwise Statistical Significance (Wilcoxon Signed-Rank Test)\n\n")
            rf.write("Tests the null hypothesis that the metric distribution medians are equal. "
                     "Significant differences ($p < 0.05$) are highlighted.\n\n")
            model_names_list = list(eval_results.keys())
            for i in range(len(model_names_list)):
                for j in range(i + 1, len(model_names_list)):
                    name_a, name_b = model_names_list[i], model_names_list[j]
                    rf.write(f"### {name_a} vs {name_b}\n\n")
                    try:
                        sig_df = significance_test(eval_results[name_a], eval_results[name_b], metric="iou")
                        rf.write(sig_df.to_markdown(index=False))
                        rf.write("\n\n")
                    except Exception as e:
                        rf.write(f"Failed to calculate Wilcoxon test: {e}\n\n")

    print(f"Report saved to: {report_file_path.resolve()}")

    # 7. Generate Visualizations
    if any(plot_ops.values()):
        print("\n" + "="*50)
        print(" GENERATING VISUALIZATION PLOTS ")
        print("="*50)

        viz = SegmentationVisualizer(class_names=CLASS_NAMES)

        if plot_ops.get("metrics_summary", True):
            for model_name, res in eval_results.items():
                print(f"Generating metrics summary for {model_name}...")
                summary_df = res.summary_df()
                fig, _ = viz.plot_metrics_summary(summary_df, title=f"Metrics Summary - {model_name}")
                viz.save_figure(fig, plots_dir / f"metrics_summary_{model_name.lower().replace('-', '_')}.png")

        if plot_ops.get("model_comparison", True) and len(eval_results) >= 2:
            print("Generating cross-model comparison plot...")
            fig, _ = viz.plot_model_comparison(comparison_df, metric="IoU",  title="Model Comparison (IoU)")
            viz.save_figure(fig, plots_dir / "model_comparison_iou.png")
            fig, _ = viz.plot_model_comparison(comparison_df, metric="Dice", title="Model Comparison (Dice)")
            viz.save_figure(fig, plots_dir / "model_comparison_dice.png")

        # Select samples with features
        max_samples        = plot_ops.get("max_panel_samples", 5)
        interesting_indices: list[int] = []
        for idx in range(len(dataset)):
            _, mask = dataset[idx]
            if mask.sum() > 50:
                interesting_indices.append(idx)
                if len(interesting_indices) >= max_samples:
                    break
        if not interesting_indices:
            interesting_indices = list(range(min(len(dataset), max_samples)))

        # Qualitative panels — one model load per name
        for model_name, res in eval_results.items():
            if model_name not in adapter_cache:
                m_cfg_match = next(c for c in model_configs if c["name"] == model_name)
                model       = load_model(m_cfg_match, device)
                model_type  = m_cfg_match.get("type", "semantic")
                a_args      = m_cfg_match.get("adapter_args", {})
                adapter_cache[model_name] = create_adapter(
                    model=model, model_name=model_name,
                    model_type=model_type, **a_args,
                )

            adapter = adapter_cache[model_name]
            effective_from_logits = adapter.output_is_logits

            if plot_ops.get("prediction_panel", True):
                print(f"Generating prediction panels for {model_name}...")
                for idx_count, idx in enumerate(interesting_indices):
                    img_tensor, mask_tensor = dataset[idx]
                    with torch.no_grad():
                        pred_logits = adapter.predict(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        probs       = torch.sigmoid(pred_logits) if effective_from_logits else pred_logits
                        pred_mask   = (probs > threshold).float().numpy()
                    fig, _ = viz.plot_prediction_panel(
                        img_tensor.numpy(), mask_tensor.numpy(), pred_mask, class_idx=0,
                        title=f"{model_name} Prediction Panel - Sample {idx_count+1}",
                    )
                    viz.save_figure(
                        fig,
                        plots_dir / f"prediction_panel_{model_name.lower().replace('-','_')}_sample_{idx_count+1}.png",
                    )

            if plot_ops.get("class_comparison", True):
                print(f"Generating class comparison grids for {model_name}...")
                for idx_count, idx in enumerate(interesting_indices):
                    img_tensor, mask_tensor = dataset[idx]
                    with torch.no_grad():
                        pred_logits = adapter.predict(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                        probs       = torch.sigmoid(pred_logits) if effective_from_logits else pred_logits
                        pred_mask   = (probs > threshold).float().numpy()
                    fig, _ = viz.plot_class_comparison(
                        img_tensor.numpy(), mask_tensor.numpy(), pred_mask,
                        title=f"{model_name} Per-Class Grid - Sample {idx_count+1}",
                    )
                    viz.save_figure(
                        fig,
                        plots_dir / f"class_comparison_{model_name.lower().replace('-','_')}_sample_{idx_count+1}.png",
                    )

        # Threshold sensitivity — use pre-computed sweep results (no re-inference)
        if plot_ops.get("threshold_sensitivity", True):
            for model_name, s_res in sweep_results.items():
                print(f"Generating threshold sensitivity plot for {model_name} (from cache)...")
                fig, _ = viz.plot_threshold_sensitivity(
                    s_res, title=f"Threshold Sensitivity - {model_name}"
                )
                viz.save_figure(
                    fig,
                    plots_dir / f"threshold_sensitivity_{model_name.lower().replace('-','_')}.png",
                )

        print(f"\nAll plots saved to: {plots_dir.resolve()}")
        print("\nEvaluation Orchestrator Run Completed successfully!")


if __name__ == "__main__":
    main()
