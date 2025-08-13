"""
Module for loading and managing ML project data.
"""

from importlib.resources import path
import warnings
from pathlib import Path    
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Class for loading and managing Kaggle competition data.
    """
    
    def __init__(self, data_dir=".", train_file="train.csv", test_file="test.csv", submission_file="sample_submission.csv"):
        """
        Initializes the DataLoader.
        
        Args:
            data_dir (str): Directory where CSV files are located
            train_file (str): Training file name
            test_file (str): Test file name
            submission_file (str): Submission file name
        """
        self.data_dir = Path(data_dir)
        self.train_file = train_file
        self.test_file = test_file
        self.submission_file = submission_file
        self.train_df = None
        self.test_df = None
        self.sample_submission_df = None
    
    def load_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Reads a CSV with safe default options."""
        return pd.read_csv(path, low_memory=False, **kwargs)

    def load_parquet(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Reads a Parquet file if you have pyarrow/fastparquet installed in your environment."""
        return pd.read_parquet(path, **kwargs)

    def load_competition_data(self, train_path: str | Path, test_path: str | Path, target_col: str = "species") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads dataset from CSVs and separates it into features and target.
        
        Returns:
        X_train, y_train, X_test, y_test
        """
        
        try:
            df_train = self.load_csv(train_path)
            df_test = self.load_csv(test_path)

            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]

            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col]

            return X_train, y_train, X_test, y_test
        except FileNotFoundError as e:
                print(f"❌ Error: Data files not found: {e}")
                print("💡 Make sure the files are in the specified directory")
                return None, None, None, None

    def load_competition_data(self, data_dir=".", train_data_file="train.csv", test_data_file="test.csv"):
        """
        Legacy function to maintain compatibility.
        """
        loader = DataLoader(data_dir, train_data_file, test_data_file)
        return loader.load_competition_data()
