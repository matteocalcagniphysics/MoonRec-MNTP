import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import measure
from scipy.spatial import distance
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importiamo le classi dal preprocessing del progetto
try:
    from lunar_segmentation.data.preprocessing import CLASS_NAMES
except ImportError:
    CLASS_NAMES = [
        'impact_crater', 'pit_skylight', 'wrinkle_ridge',
        'lobate_scarp', 'irregular_mare_patch', 'apollo_site', 'candidate_rille'
    ]

class DatasetAnalyzer:
    """
    Classe per l'analisi statistica e la pulizia del dataset lunare.
    Calcola metriche sulle immagini (radiometria/contrasto) e sulle maschere (morfologia, densità).
    """
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.stats_list = []

    def compute_nearest_neighbor_distance(self, centroids):
        """Calcola la distanza media del primo vicino (Nearest Neighbor) per un set di centroidi."""
        if len(centroids) < 2:
            return np.nan
        # Calcola la matrice delle distanze a coppie
        dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
        # Sostituisce la diagonale (distanza da se stessi) con infinito
        np.fill_diagonal(dist_matrix, np.inf)
        # Trova la distanza minima per ogni punto
        min_distances = dist_matrix.min(axis=1)
        return min_distances.mean()

    def analyze_tile(self, tile_path: Path):
        """Analizza un singolo file .npz contenente image e mask."""
        data = np.load(tile_path)
        
        # Le immagini nel progetto sono salvate in formato (C, H, W)
        # image: (3, 128, 128), mask: (7, 128, 128)
        image = data['image']
        mask = data['mask']
        
        # 1. Analisi Radiometrica dell'Immagine (come da Slide: illuminazione, contrasto)
        # Il canale 0 è la normalizzazione base, il canale 1 è CLAHE, il 2 è Sobel
        mean_brightness = image[0].mean()
        contrast_std = image[0].std()
        
        tile_stats = {
            'tile_name': tile_path.name,
            'mean_brightness': mean_brightness,
            'contrast_std': contrast_std,
            'is_empty_image': contrast_std < 1e-4  # Immagine piatta/nera
        }
        
        # 2. Analisi Morfologica delle Maschere
        total_positive_pixels = 0
        
        for i, class_name in enumerate(CLASS_NAMES):
            class_mask = mask[i]
            
            # Quanti pixel positivi ci sono per questa classe? (Densità)
            area = class_mask.sum()
            total_positive_pixels += area
            
            # Etichetta le componenti connesse (conta gli oggetti, es. quanti crateri?)
            labeled_mask, num_features = measure.label(class_mask > 0, return_num=True)
            
            nn_distance = np.nan
            if num_features > 0:
                # Calcola i centroidi di ogni oggetto
                props = measure.regionprops(labeled_mask)
                centroids = np.array([prop.centroid for prop in props])
                
                # Calcola la distanza dei primi vicini se c'è più di un oggetto
                if num_features > 1:
                    nn_distance = self.compute_nearest_neighbor_distance(centroids)
            
            # Salvataggio metriche specifiche per classe
            tile_stats[f'{class_name}_area'] = area
            tile_stats[f'{class_name}_count'] = num_features
            tile_stats[f'{class_name}_nn_dist'] = nn_distance

        tile_stats['total_mask_area'] = total_positive_pixels
        self.stats_list.append(tile_stats)

    def process_directory(self):
        """Itera su tutti i file .npz nella directory e crea un DataFrame."""
        tile_files = list(self.data_dir.rglob('*.npz'))
        logger.info(f"Trovati {len(tile_files)} tiles. Inizio analisi...")
        
        for file_path in tile_files:
            self.analyze_tile(file_path)
            
        self.df = pd.DataFrame(self.stats_list)
        logger.info("Analisi completata.")
        return self.df

    def plot_statistics(self, output_dir: Path):
        """Genera e salva istogrammi e grafici per l'analisi QA del dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="darkgrid")

        # 1. Distribuzione del Contrasto (per trovare immagini "bruciate" o in ombra totale)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='contrast_std', bins=50, kde=True, color='purple')
        plt.title('Distribuzione del Contrasto delle Immagini (Std Dev)')
        plt.xlabel('Deviazione Standard dei Pixel')
        plt.ylabel('Frequenza')
        plt.axvline(x=0.05, color='red', linestyle='--', label='Soglia Basso Contrasto Suggerita')
        plt.legend()
        plt.savefig(output_dir / 'image_contrast_distribution.png')
        plt.close()

        # 2. Quante classi compaiono nei tiles? (Oggetti totali)
        count_cols = [f'{c}_count' for c in CLASS_NAMES]
        counts_df = self.df[count_cols].sum().reset_index()
        counts_df.columns = ['Classe', 'Numero Totale di Oggetti']
        counts_df['Classe'] = counts_df['Classe'].str.replace('_count', '')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=counts_df, x='Classe', y='Numero Totale di Oggetti', palette='viridis')
        plt.title('Conteggio Totale degli Oggetti per Classe nel Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'class_object_counts.png')
        plt.close()

        # 3. Istogramma della densità dei crateri (Area totale per tile)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df[self.df['impact_crater_area'] > 0], x='impact_crater_area', bins=50, color='blue')
        plt.title('Distribuzione dell\'Area Totale dei Crateri (per Tile)')
        plt.xlabel('Area (Pixel positivi)')
        plt.ylabel('Numero di Tiles')
        plt.savefig(output_dir / 'crater_area_distribution.png')
        plt.close()

        # 4. Nearest Neighbor Distance (Quanto sono raggruppati gli oggetti?)
        nn_cols = [f'{c}_nn_dist' for c in CLASS_NAMES]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[nn_cols], orient='h', palette='magma')
        plt.title('Distanza Media dei Primi Vicini (Clustering degli oggetti)')
        plt.xlabel('Distanza (pixel)')
        plt.yticks(ticks=range(len(CLASS_NAMES)), labels=CLASS_NAMES)
        plt.tight_layout()
        plt.savefig(output_dir / 'nearest_neighbor_distances.png')
        plt.close()
        
        logger.info(f"Grafici salvati in {output_dir}")

    def filter_dataset(self, min_contrast: float = 0.02, require_mask: bool = True):
        """
        Filtra i tiles inutili.
        Rimuove immagini senza contrasto (es. zone in ombra totale ai poli).
        Opzionalmente rimuove immagini che non hanno alcuna label (background puro).
        """
        initial_len = len(self.df)
        
        # Filtro 1: Rimuovi immagini troppo scure/piatte (Radiometria fisica)
        valid_df = self.df[self.df['contrast_std'] >= min_contrast]
        
        # Filtro 2: Rimuovi immagini vuote (Background puro) se richiesto
        if require_mask:
            valid_df = valid_df[valid_df['total_mask_area'] > 0]
            
        final_len = len(valid_df)
        logger.info(f"Filtraggio completato: mantenuti {final_len}/{initial_len} tiles ({(final_len/initial_len)*100:.1f}%)")
        
        # Restituisce l'elenco dei nomi dei file considerati validi
        return valid_df['tile_name'].tolist()


# ======================================================================== #
#  Test Syntetico per dimostrazione
# ======================================================================== #
if __name__ == "__main__":
    from lunar_segmentation.evaluation.synthetic import generate_realistic_batch
    import torch
    
    logger.info("Avvio test con dati sintetici...")
    # Creiamo una cartella temporanea con finti .npz per testare il codice
    test_dir = Path("temp_eda_test")
    test_dir.mkdir(exist_ok=True)
    
    # Generiamo un batch sintetico (B, C, H, W)
    images, masks, _ = generate_realistic_batch(batch_size=10, height=128, width=128)
    
    for b in range(10):
        np.savez_compressed(
            test_dir / f"synthetic_tile_{b}.npz", 
            image=images[b].numpy(), 
            mask=masks[b].numpy()
        )
        
    analyzer = DatasetAnalyzer(test_dir)
    df = analyzer.process_directory()
    analyzer.plot_statistics(test_dir / "plots")
    
    valid_tiles = analyzer.filter_dataset(min_contrast=0.01, require_mask=True)
    print(f"\nEsempio tiles validi: {valid_tiles[:3]}")
    
    logger.info("Test completato. Controlla la cartella 'temp_eda_test/plots'")