import numpy as np
import pandas as pd
from pathlib import Path
from rasterio import features
from rasterio.transform import Affine
from skimage import exposure, filters, morphology
import torch
import logging

logger = logging.getLogger(__name__)

# Default classes as used in the notebook for multi-label segmentation
CLASS_NAMES = [
    'impact_crater',
    'pit_skylight',
    'wrinkle_ridge',
    'lobate_scarp',
    'irregular_mare_patch',
    'apollo_site',
    'candidate_rille'
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

def build_three_channel_input(gray: np.ndarray) -> np.ndarray:
    """
    Creates a 3-channel input for the model using normalization, CLAHE, and Sobel filtering.
    """
    gray = gray.astype(np.float32)
    if gray.size == 0:
        return np.zeros((3, gray.shape[0], gray.shape[1]), dtype=np.float32)

    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        norm = (gray - gmin) / (gmax - gmin)
    else:
        norm = np.zeros_like(gray, dtype=np.float32)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = exposure.equalize_adapthist(norm, clip_limit=0.03).astype(np.float32)
    # Sobel operator for edge detection
    sobel = filters.sobel(norm).astype(np.float32)

    return np.stack([norm, clahe, sobel], axis=0)

def rasterize_multilabel(labels: dict, out_shape, transform, raster_crs):
    """
    Rasterize all label GeoDataFrames into a multi-channel uint8 mask.

    - impact_crater: Circular polygons — NO dilation needed (already area features)
    - candidate_rille: Polygon features — moderate dilation only
    - pit_skylight, apollo_site: Point features — disk(5) dilation
    - wrinkle_ridge, lobate_scarp: Line features — disk(3) dilation
    - irregular_mare_patch: Area or point features — disk(5) dilation
    """
    mask = np.zeros((len(CLASS_NAMES), out_shape[0], out_shape[1]), dtype=np.uint8)

    for class_name, gdf in labels.items():
        if class_name not in CLASS_TO_INDEX or gdf is None or len(gdf) == 0:
            continue
        idx = CLASS_TO_INDEX[class_name]

        # Reproject to raster CRS
        try:
            if gdf.crs is not None and raster_crs is not None:
                layer = gdf.to_crs(raster_crs)
            else:
                layer = gdf
        except Exception as e:
            logger.warning(f"CRS reprojection failed for {class_name}: {e}, using raw coordinates")
            layer = gdf

        shapes_iter = [(geom, 1) for geom in layer.geometry if geom is not None and not geom.is_empty]
        if not shapes_iter:
            logger.info(f"No valid geometries for {class_name} in this extent")
            continue

        channel = features.rasterize(
            shapes=shapes_iter,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8',
        )

        positive_before = int(channel.sum())

        # Apply class-specific dilation
        if class_name == 'impact_crater':
            # Craters are already circular polygons — no dilation needed.
            # The Polygon geometries already encode the full crater diameter.
            pass
        elif class_name == 'candidate_rille':
            # Rilles may be polygons already; apply light dilation to fill gaps
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(2)).astype(np.uint8)
        elif class_name in {'pit_skylight', 'apollo_site'}:
            # Point features: expand to visible area
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(5)).astype(np.uint8)
        elif class_name in {'wrinkle_ridge', 'lobate_scarp'}:
            # Line features: thicken traces
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(3)).astype(np.uint8)
        elif class_name == 'irregular_mare_patch':
            # Area / point features: expand
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(5)).astype(np.uint8)

        positive_after = int(channel.sum())
        logger.info(f"  {class_name}: {positive_before} → {positive_after} positive pixels "
                    f"(dilation applied: {positive_before != positive_after})")

        mask[idx] = channel
    return mask

def iter_tile_origins(height: int, width: int, tile_size: int, stride: int):
    """
    Generator for tiling coordinates.
    """
    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))

    if rows and rows[-1] != height - tile_size:
        rows.append(max(height - tile_size, 0))
    if cols and cols[-1] != width - tile_size:
        cols.append(max(width - tile_size, 0))

    for r in rows:
        for c in cols:
            yield r, c

def save_tiles_for_aoi(aoi_name: str, bounds, raster_path: Path, labels: dict,
                     tile_size: int = 256, stride: int = 128,
                     processed_dir: Path = Path("data/processed"),
                     keep_empty_fraction: float = 0.1, seed: int = 42):
    """
    Tiles a lunar raster and its labels, saving them as .npz files.
    """
    from ..utils.geo_utils import crop_singleband_raster

    img_gray, transform, profile = crop_singleband_raster(raster_path, bounds)
    if img_gray.size == 0 or img_gray.shape[0] < tile_size or img_gray.shape[1] < tile_size:
        logger.warning(f"AOI {aoi_name}: raster crop too small {img_gray.shape}, skipping.")
        return pd.DataFrame(columns=['aoi', 'tile_path', 'row', 'col', 'positive_pixels'])

    raster_crs = profile.get('crs')
    image = build_three_channel_input(img_gray)
    mask = rasterize_multilabel(labels, img_gray.shape, transform, raster_crs)

    tile_dir = processed_dir / 'tiles' / aoi_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    records = []
    keep_rng = np.random.default_rng(seed)

    for r, c in iter_tile_origins(img_gray.shape[0], img_gray.shape[1], tile_size, stride):
        x = image[:, r:r+tile_size, c:c+tile_size]
        y = mask[:, r:r+tile_size, c:c+tile_size]

        if x.shape[1] != tile_size or x.shape[2] != tile_size:
            continue

        positive = int(y.sum())
        if positive == 0 and keep_rng.random() > keep_empty_fraction:
            continue

        tile_path = tile_dir / f'{aoi_name}_r{r:05d}_c{c:05d}.npz'
        np.savez_compressed(tile_path, image=x.astype(np.float32), mask=y.astype(np.uint8),
                             row=r, col=c, transform=np.array(transform))

        records.append({
            'aoi': aoi_name,
            'tile_path': str(tile_path),
            'row': r,
            'col': c,
            'positive_pixels': positive,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Added functions — tile QA, spatial split, class distribution
# ---------------------------------------------------------------------------

def is_valid_tile(image: np.ndarray, stripe_threshold: float = 2.0) -> bool:
    """
    Returns False if the tile has horizontal stripe artifacts.

    Stripes cause high column variance relative to row variance.
    Threshold 2.0 was calibrated on slide 23 examples:
      rejected tiles (r00000_c00052/55/60) scored 2.76-4.12,
      valid tile (r00074_c00170) scored 1.02.

    Parameters
    ----------
    image            : (C, H, W) tile image
    stripe_threshold : col_var / row_var ratio above which tile is rejected
    """
    gray = image[0].astype(np.float32)
    row_var = float(np.var(gray, axis=1).mean())
    col_var = float(np.var(gray, axis=0).mean())
    stripe_score = col_var / (row_var + 1e-8)
    is_valid = stripe_score < stripe_threshold
    logger.debug(f"stripe_score={stripe_score:.2f} → {'valid' if is_valid else 'rejected'}")
    return is_valid


def spatial_train_val_split(
    index_df: pd.DataFrame,
    val_fraction: float = 0.2,
    split_axis: str = 'row',
) -> tuple:
    """
    Split tiles by geographic position to avoid data leakage.

    A random split leaks information because adjacent tiles share up to 50%
    of their pixels (stride=128, tile_size=256). Splitting on row or col
    ensures train and val tiles come from different regions of the mosaic.

    Parameters
    ----------
    index_df     : DataFrame with columns row, col, tile_path, ...
    val_fraction : fraction of tiles for validation (default 0.2)
    split_axis   : 'row' splits horizontally, 'col' vertically

    Returns
    -------
    train_df, val_df
    """
    threshold = int(np.quantile(index_df[split_axis].values, 1 - val_fraction))
    train_df = index_df[index_df[split_axis] < threshold].reset_index(drop=True)
    val_df   = index_df[index_df[split_axis] >= threshold].reset_index(drop=True)
    logger.info(
        f"Spatial split on '{split_axis}' at {threshold}: "
        f"{len(train_df)} train / {len(val_df)} val tiles"
    )
    return train_df, val_df


def compute_class_distribution(
    index_df: pd.DataFrame,
    tile_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Per-class tile counts and pixel counts across the dataset.

    Used to compute the class_weights in unet_config.yaml:
    suggested_weight = 100 / freq_percent.

    Parameters
    ----------
    index_df : DataFrame from index.csv
    tile_dir : optional base directory for relative tile paths

    Returns
    -------
    DataFrame: class, n_tiles, total_pixels, freq_percent, suggested_weight
    """
    counts  = {name: 0 for name in CLASS_NAMES}
    n_tiles = {name: 0 for name in CLASS_NAMES}
    total   = 0

    for _, row in index_df.iterrows():
        path = Path(row['tile_path'])
        if tile_dir is not None and not path.is_absolute():
            path = tile_dir / path
        if not path.exists():
            continue
        data = np.load(path)
        mask = data['mask']
        total += 1
        for c, name in enumerate(CLASS_NAMES):
            if mask[c].sum() > 0:
                n_tiles[name] += 1
                counts[name]  += int(mask[c].sum())

    records = []
    for name in CLASS_NAMES:
        freq = n_tiles[name] / max(total, 1) * 100
        records.append({
            'class':            name,
            'n_tiles':          n_tiles[name],
            'total_pixels':     counts[name],
            'freq_percent':     round(freq, 2),
            'suggested_weight': round(100.0 / max(freq, 0.01), 1),
        })

    logger.info(f"Class distribution computed on {total} tiles.")
    return pd.DataFrame(records)
    
    
    
    
# =============================================================================
# OPTIONAL BULK PREPROCESSING SCRIPT (COMMENTED OUT FOR SAFETY)
# To execute this pipeline, uncomment the block below and run: python preprocessing.py
# =============================================================================
# if __name__ == "__main__":
#     import geopandas as gpd
#     
#     # 1. Setup local data paths using Path.home() for portability
#     BASE_DATA_DIR = Path.home() / 'MoonRec-MNTP' / 'data' / 'MR'
#     RASTER_PATH = BASE_DATA_DIR / 'lunar_mosaic.tif'
#     
#     # Define isolated directory for processed outputs to protect raw data
#     NEW_PROCESSED_DIR = BASE_DATA_DIR / 'processed_dataset'
#     TILES_OUTPUT_DIR = NEW_PROCESSED_DIR / 'tiles'
#     
#     TILES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     
#     print("Paths configured:")
#     print(f" Raw data source: {BASE_DATA_DIR}")
#     print(f" Processed output target: {NEW_PROCESSED_DIR}\n")
# 
#     # 2. Load vector spatial layers (GeoJSON) - Requires geopandas
#     print("Loading GeoJSON files...")
#     labels_dict = {}
#     
#     geojson_files = {
#         'impact_crater': BASE_DATA_DIR / 'craters.geojson',
#         'wrinkle_ridge': BASE_DATA_DIR / 'ridges.geojson',
#         # Add additional class layers here if required
#     }
#     
#     for class_name, file_path in geojson_files.items():
#         if file_path.exists():
#             labels_dict[class_name] = gpd.read_file(file_path)
#             print(f" Loaded {class_name} ({len(labels_dict[class_name])} features)")
#         else:
#             print(f" Warning: {file_path} not found. Skipping class '{class_name}'.")
#     
#     # 3. Execute processing pipeline
#     print("\nExecuting bulk preprocessing...")
#     
#     if not RASTER_PATH.exists():
#         print(f" Error: Lunar mosaic TIF not found at {RASTER_PATH}")
#     else:
#         index_df = save_tiles_for_aoi(
#             aoi_name="marius_hills",
#             bounds=(0, 0, 10000, 10000),
#             raster_path=RASTER_PATH,
#             labels=labels_dict,
#             processed_dir=TILES_OUTPUT_DIR
#         )
#         
#         # Save generated dataset index to output directory
#         csv_final_path = NEW_PROCESSED_DIR / "index.csv"
#         index_df.to_csv(csv_final_path, index=False)
#         
#         print(f"\nPreprocessing finished.")
#         print(f" Tiles generated in: {TILES_OUTPUT_DIR}")
#         print(f" Index saved to: {csv_final_path}")
#         print(" Verify 'processed_dataset/' is added to .gitignore before pushing.")
