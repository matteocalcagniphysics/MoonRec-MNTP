# Data Module

This module manages the entire data lifecycle: from resolving remote scientific data sources to providing high-performance PyTorch datasets for training.

## Key Features

- **Automated Ingestion**: `resolver.py` manages a manifest of remote URLs (LROC, USGS, NASA) and handles automatic downloading and extraction.
- **Heterogeneous Label Loading**: `label_loader.py` unifies diverse spatial formats (Shapefiles, GeoPackages, CSVs, Excel) into a standard lunar geographic CRS.
- **Advanced Preprocessing**: `preprocessing.py` implements a 3-channel input strategy (Normalization + CLAHE + Sobel) and multi-label rasterization with class-specific morphological dilation.
- **Data Augmentation**: `datasets.py` provides a `MoonTileDataset` with on-the-fly geometric augmentations (flips, rotations).
- **Dataset QA (EDA)**: `exploration_eda.py` provides automated quality assurance, analyzing image contrast and label density to prune the dataset of non-informative tiles.

## File Overview

- `resolver.py`: Handles remote data resolution and caching.
- `label_loader.py`: Loads vector features and handles crater diameter-to-polygon conversions.
- `preprocessing.py`: Implements image normalization, tiling logic, and mask rasterization.
- `datasets.py`: PyTorch `Dataset` implementation for loading tiled `.npz` files.
- `exploration_eda.py`: Statistical analysis and dataset cleaning tools.

## Data Pipeline Flow

1. **Resolve**: `resolver.prepare_dataset("data/")` downloads raw imagery and vector files.
2. **Process**: Vector labels are loaded via `label_loader` and rasterized into tiles using `preprocessing.save_tiles_for_aoi`.
3. **Analyze**: `exploration_eda.DatasetAnalyzer` runs QA on processed tiles to filter low-quality data.
4. **Load**: `MoonTileDataset` feeds tiles into the PyTorch training loop.

## CRS & Coordinate Handling
All spatial operations use a unified Lunar Geographic CRS (0-360° or -180/180° longitude, -90/90° latitude). The `label_loader` ensures that all vector features are correctly reprojected to match the raster geometry during rasterization.

## Usage Example: EDA Analysis
To run Quality Assurance on your processed tiles:

```python
from lunar_segmentation.data.exploration_eda import DatasetAnalyzer

analyzer = DatasetAnalyzer("data/processed/tiles")
df = analyzer.process_directory()
analyzer.plot_statistics("data/reports/eda")

# Filter tiles with low contrast (e.g., polar shadows)
valid_tiles = analyzer.filter_dataset(min_contrast=0.05)
```
