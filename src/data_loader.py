import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class LyricsDataLoader:
    def __init__(self, data_dir='csv'):
        self.data_dir = Path(data_dir)
        
    def load_all_artists(self):
        """Load all artist CSV files and combine them into one dataframe"""
        all_files = glob.glob(str(self.data_dir / "*.csv"))
        dfs = []
        
        for file in all_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
        if not dfs:
            raise ValueError("No CSV files found in the data directory")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    def load_single_artist(self, artist_name):
        """Load data for a specific artist"""
        file_path = self.data_dir / f"{artist_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for artist: {artist_name}")
            
        return pd.read_csv(file_path)
    
    def get_available_artists(self):
        """Get list of all available artists in the dataset"""
        files = glob.glob(str(self.data_dir / "*.csv"))
        return [Path(f).stem for f in files] 