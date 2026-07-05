import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocess")

# Data Paths
INDEX_CSV      = Path("/LCP/MoonRec-MNTP/data/MR/tiles/index.csv")
OUTPUT_DIR     = Path("/LCP/MoonRec-MNTP/data/MR/tiles")
TILES_BASE_DIR = Path("/LCP/MoonRec-MNTP/data/MR")

# Split parameters into training and validation set
VAL_FRACTION   = 0.2   # 20% of tiles go to validation
SPLIT_AXIS     = "row" # split horizontally (rows); use "col" for vertical
FILTER_STRIPES = True  # set to False to skip the stripe-artifact filter

# Import the required function from preprocessing and split the dataset
from preprocessing import get_train_val_split

logger.info("Reading index: %s", INDEX_CSV)

if not INDEX_CSV.exists():
    logger.error("index.csv not found at %s", INDEX_CSV)
    sys.exit(1)

n_total = len(pd.read_csv(INDEX_CSV))
logger.info("Total tiles in index: %d", n_total)

logger.info(
    "Running get_train_val_split (filter_stripes=%s, split_axis='%s', val_fraction=%.0f%%)",
    FILTER_STRIPES, SPLIT_AXIS, VAL_FRACTION * 100,
)

train_df, val_df = get_train_val_split(
    index_csv=INDEX_CSV,
    base_dir=TILES_BASE_DIR,   # resolves relative tile_path entries in index.csv
    val_fraction=VAL_FRACTION,
    split_axis=SPLIT_AXIS,
    filter_stripes=FILTER_STRIPES,
)

n_kept = len(train_df) + len(val_df)
logger.info(
    "Result: %d kept / %d discarded (stripes+missing) → %d train / %d val",
    n_kept, n_total - n_kept, len(train_df), len(val_df),
)

if len(train_df) == 0:
    logger.error("No training tiles remaining. Check INDEX_CSV path and tile files.")
    sys.exit(1)

# Save train and val csv files to be used for training
train_csv = OUTPUT_DIR / "train.csv"
val_csv   = OUTPUT_DIR / "val.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv,     index=False)

logger.info("Saved: %s  (%d rows)", train_csv, len(train_df))
logger.info("Saved: %s  (%d rows)", val_csv,   len(val_df))

# Quick class-distribution summary
logger.info("-" * 50)
logger.info("Class distribution in train split:")

try:
    from preprocessing import compute_class_distribution
    dist_df = compute_class_distribution(train_df, tile_dir=TILES_BASE_DIR)
    logger.info("\n%s", dist_df.to_string(index=False))
except Exception as exc:
    logger.warning("Could not compute class distribution: %s", exc)

logger.info("-" * 50)
logger.info("Preprocessing complete. You can now use train.csv / val.csv for training.")
