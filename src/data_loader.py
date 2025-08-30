"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial preprocessing of cryptocurrency data"""
    
    def __init__(self, data_path: str = "gq-implied-volatility-forecasting", 
                 max_rows: Optional[int] = 50000):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to data directory
            max_rows: Maximum rows to load (None for all data)
        """
        self.data_path = Path(data_path)
        self.max_rows = max_rows
        self.crypto_symbols = ['ETH', 'BTC', 'DOGE', 'DOT', 'LINK', 'SHIB', 'SOL']
        
    def load_data(self, symbol: str = 'ETH') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data for specified cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol to load
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info(f"Loading {symbol} data...")
        
        # Load training data
        train_path = self.data_path / "train" / f"{symbol}.csv"
        test_path = self.data_path / "test" / f"{symbol}.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
            
        try:
            train_data = pd.read_csv(train_path, nrows=self.max_rows)
            test_data = pd.read_csv(test_path, nrows=self.max_rows)
            
            logger.info(f"Loaded {symbol} - Train: {train_data.shape}, Test: {test_data.shape}")
            
            # Basic data validation
            self._validate_data(train_data, test_data)
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading {symbol} data: {e}")
            raise
    
    def _validate_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Validate loaded data"""
        required_cols = ['timestamp', 'mid_price', 'bid_price1', 'ask_price1']
        
        for col in required_cols:
            if col not in train_data.columns:
                raise ValueError(f"Missing required column in training data: {col}")
            if col not in test_data.columns:
                raise ValueError(f"Missing required column in test data: {col}")
        
        # Check for target column in training data
        if 'label' not in train_data.columns:
            raise ValueError("Missing target column 'label' in training data")
        
        logger.info("✅ Data validation passed")
    
    def get_data_info(self, data: pd.DataFrame) -> dict:
        """Get basic information about the dataset"""
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        if 'label' in data.columns:
            info['target_stats'] = {
                'mean': data['label'].mean(),
                'std': data['label'].std(),
                'min': data['label'].min(),
                'max': data['label'].max()
            }
        
        return info
