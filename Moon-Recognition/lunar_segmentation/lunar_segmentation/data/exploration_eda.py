import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import measure
from scipy.spatial import distance
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import classes from project preprocessing
try:
    from lunar_segmentation.data.preprocessing import CLASS_NAMES
except ImportError:
    CLASS_NAMES = [
        'impact_crater', 'pit_skylight', 'wrinkle_ridge',
        'lobate_scarp', 'irregular_mare_patch', 'apollo_site', 'candidate_rille'
    ]

class DatasetAnalyzer:
    """
    Class for statistical analysis and cleaning of the lunar dataset.
    Computes metrics on images (radiometry/contrast) and masks (morphology, density).
    """
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.stats_list = []

    def compute_nearest_neighbor_distance(self, centroids):
        """Computes the mean nearest neighbor distance for a set of centroids."""
        if len(centroids) < 2:
            return np.nan
        # Compute pairwise distance matrix
        dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
        # Replace diagonal (distance to self) with infinity
        np.fill_diagonal(dist_matrix, np.inf)
        # Find minimum distance for each point
        min_distances = dist_matrix.min(axis=1)
        return min_distances.mean()

    def analyze_tile(self, tile_path: Path):
        """Analyzes a single .npz file containing image and mask."""
        data = np.load(tile_path)
        
        # Images in the project are saved in (C, H, W) format
        # image: (3, 128, 128), mask: (7, 128, 128)
        image = data['image']
        mask = data['mask']
        
        # 1. Image Radiometric Analysis (as per slides: illumination, contrast)
        # Channel 0 is base normalization, channel 1 is CLAHE, channel 2 is Sobel
        mean_brightness = image[0].mean()
        contrast_std = image[0].std()
        
        tile_stats = {
            'tile_name': tile_path.name,
            'mean_brightness': mean_brightness,
            'contrast_std': contrast_std,
            'is_empty_image': contrast_std < 1e-4  # Flat/black image
        }
        
        # 2. Mask Morphological Analysis
        total_positive_pixels = 0
        
        for i, class_name in enumerate(CLASS_NAMES):
            class_mask = mask[i]
            
            # How many positive pixels are there for this class? (Density)
            area = class_mask.sum()
            total_positive_pixels += area
            
            # Label connected components (count objects, e.g. how many craters?)
            labeled_mask, num_features = measure.label(class_mask > 0, return_num=True)
            
            nn_distance = np.nan
            if num_features > 0:
                # Compute centroids of each object
                props = measure.regionprops(labeled_mask)
                centroids = np.array([prop.centroid for prop in props])
                
                # Compute nearest neighbor distance if there is more than one object
                if num_features > 1:
                    nn_distance = self.compute_nearest_neighbor_distance(centroids)
            
            # Save class-specific metrics
            tile_stats[f'{class_name}_area'] = area
            tile_stats[f'{class_name}_count'] = num_features
            tile_stats[f'{class_name}_nn_dist'] = nn_distance

        tile_stats['total_mask_area'] = total_positive_pixels
        self.stats_list.append(tile_stats)

    def process_directory(self):
        """Iterates over all .npz files in the directory and creates a DataFrame."""
        tile_files = list(self.data_dir.rglob('*.npz'))
        logger.info(f"Found {len(tile_files)} tiles. Starting analysis...")
        
        for file_path in tile_files:
            self.analyze_tile(file_path)
            
        self.df = pd.DataFrame(self.stats_list)
        logger.info("Analysis completed.")
        return self.df

    def plot_statistics(self, output_dir: Path):
        """Generates and saves histograms and plots for QA analysis of the dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="darkgrid")

        # 1. Contrast Distribution (to find overexposed or deep shadow images)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='contrast_std', bins=50, kde=True, color='purple')
        plt.title('Image Contrast Distribution (Std Dev)')
        plt.xlabel('Pixel Standard Deviation')
        plt.ylabel('Frequency')
        plt.axvline(x=0.05, color='red', linestyle='--', label='Suggested Low Contrast Threshold')
        plt.legend()
        plt.savefig(output_dir / 'image_contrast_distribution.png')
        plt.close()

        # 2. How many classes appear in tiles? (Total objects)
        count_cols = [f'{c}_count' for c in CLASS_NAMES]
        counts_df = self.df[count_cols].sum().reset_index()
        counts_df.columns = ['Class', 'Total Number of Objects']
        counts_df['Class'] = counts_df['Class'].str.replace('_count', '')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=counts_df, x='Class', y='Total Number of Objects', palette='viridis')
        plt.title('Total Object Count per Class in Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'class_object_counts.png')
        plt.close()

        # 3. Crater density histogram (Total area per tile)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df[self.df['impact_crater_area'] > 0], x='impact_crater_area', bins=50, color='blue')
        plt.title('Total Crater Area Distribution (per Tile)')
        plt.xlabel('Area (Positive pixels)')
        plt.ylabel('Number of Tiles')
        plt.savefig(output_dir / 'crater_area_distribution.png')
        plt.close()

        # 4. Nearest Neighbor Distance (How clustered are the objects?)
        nn_cols = [f'{c}_nn_dist' for c in CLASS_NAMES]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[nn_cols], orient='h', palette='magma')
        plt.title('Mean Nearest Neighbor Distance (Object clustering)')
        plt.xlabel('Distance (pixels)')
        plt.yticks(ticks=range(len(CLASS_NAMES)), labels=CLASS_NAMES)
        plt.tight_layout()
        plt.savefig(output_dir / 'nearest_neighbor_distances.png')
        plt.close()
        
        logger.info(f"Plots saved in {output_dir}")

    def filter_dataset(self, min_contrast: float = 0.02, require_mask: bool = True):
        """
        Filters out useless tiles.
        Removes images with no contrast (e.g. zones in complete shadow at the poles).
        Optionally removes images that do not have any label (pure background).
        """
        initial_len = len(self.df)
        
        # Filter 1: Remove images that are too dark/flat (physical radiometry)
        valid_df = self.df[self.df['contrast_std'] >= min_contrast]
        
        # Filter 2: Remove empty images (pure background) if requested
        if require_mask:
            valid_df = valid_df[valid_df['total_mask_area'] > 0]
            
        final_len = len(valid_df)
        logger.info(f"Filtering completed: kept {final_len}/{initial_len} tiles ({(final_len/initial_len)*100:.1f}%)")
        
        # Returns the list of valid file names
        return valid_df['tile_name'].tolist()

