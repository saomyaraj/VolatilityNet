"""
Feature engineering module for order book and technical indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles comprehensive feature engineering from order book data"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 10):
        """
        Initialize FeatureEngineer
        
        Args:
            sequence_length: Length of input sequences in seconds
            prediction_horizon: Prediction horizon in seconds
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.selected_features = None
        self.scaler = None
        self.target_scaler = None
        
    def create_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Create comprehensive features from raw data
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            
        Returns:
            Tuple of (train_features_dict, test_features_dict)
        """
        logger.info("Creating order book features...")
        train_features = self._create_order_book_features(train_data.copy())
        test_features = self._create_order_book_features(test_data.copy())
        
        logger.info("Creating technical indicators...")
        train_features = self._create_technical_features(train_features)
        test_features = self._create_technical_features(test_features)
        
        logger.info("Creating lagged features...")
        train_features = self._create_lag_features(train_features, target_col='label')
        test_features = self._create_lag_features(test_features, target_col=None)
        
        logger.info("Selecting top features...")
        self._select_features(train_features)
        
        logger.info("Preparing sequences...")
        train_sequences = self._prepare_sequences(train_features, target_col='label')
        test_sequences = self._prepare_sequences(test_features, target_col=None)
        
        # Save processed features
        self._save_processed_features(train_sequences, test_sequences)
        
        return train_sequences, test_sequences
    
    def _create_order_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from order book data"""
        # Basic price features
        df['spread'] = df['ask_price1'] - df['bid_price1']
        df['spread_pct'] = df['spread'] / df['mid_price']
        df['mid_price_return'] = df['mid_price'].pct_change()
        
        # Order book imbalance features
        bid_cols = ['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']
        ask_cols = ['ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5']
        
        df['total_bid_volume'] = df[bid_cols].sum(axis=1)
        df['total_ask_volume'] = df[ask_cols].sum(axis=1)
        df['volume_imbalance'] = (df['total_bid_volume'] - df['total_ask_volume']) / (df['total_bid_volume'] + df['total_ask_volume'])
        
        # Weighted prices
        bid_price_cols = ['bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5']
        ask_price_cols = ['ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']
        
        df['weighted_bid_price'] = (df[bid_price_cols] * df[bid_cols]).sum(axis=1) / df['total_bid_volume']
        df['weighted_ask_price'] = (df[ask_price_cols] * df[ask_cols]).sum(axis=1) / df['total_ask_volume']
        df['weighted_mid_price'] = (df['weighted_bid_price'] + df['weighted_ask_price']) / 2
        
        # Price impact features
        df['price_impact_bid'] = (df['bid_price1'] - df['bid_price5']) / df['mid_price']
        df['price_impact_ask'] = (df['ask_price5'] - df['ask_price1']) / df['mid_price']
        
        # Volume-weighted spread
        df['vw_spread'] = (df['weighted_ask_price'] - df['weighted_bid_price']) / df['mid_price']
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create technical analysis features"""
        for window in windows:
            # Moving averages
            df[f'ma_{window}'] = df['mid_price'].rolling(window=window).mean()
            df[f'price_ma_ratio_{window}'] = df['mid_price'] / df[f'ma_{window}']
            
            # Volatility
            df[f'volatility_{window}'] = df['mid_price_return'].rolling(window=window).std()
            
            # Volume features
            df[f'volume_ma_{window}'] = df['total_bid_volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['total_bid_volume'] / df[f'volume_ma_{window}']
            
            # Momentum
            df[f'momentum_{window}'] = df['mid_price'] / df['mid_price'].shift(window) - 1
            
            # Spread features
            df[f'spread_ma_{window}'] = df['spread_pct'].rolling(window=window).mean()
            df[f'spread_ratio_{window}'] = df['spread_pct'] / df[f'spread_ma_{window}']
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, target_col: str = None, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features"""
        for lag in lags:
            df[f'mid_price_lag_{lag}'] = df['mid_price'].shift(lag)
            df[f'return_lag_{lag}'] = df['mid_price_return'].shift(lag)
            df[f'spread_lag_{lag}'] = df['spread_pct'].shift(lag)
            df[f'volume_imbalance_lag_{lag}'] = df['volume_imbalance'].shift(lag)
            
            if target_col and target_col in df.columns:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def _select_features(self, train_data: pd.DataFrame, top_n: int = 50):
        """Select top features based on correlation with target"""
        feature_cols = [col for col in train_data.columns if col not in ['timestamp', 'label']]
        numeric_features = train_data[feature_cols].select_dtypes(include=[np.number]).columns
        
        correlations = []
        for col in numeric_features:
            if not train_data[col].isna().all():
                corr = train_data[col].corr(train_data['label'])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [feat[0] for feat in correlations[:top_n]]
        
        # Filter out target-related features for consistency
        self.selected_features = [f for f in selected_features if 'label' not in f.lower()]
        
        logger.info(f"Selected {len(self.selected_features)} features")
    
    def _prepare_sequences(self, df: pd.DataFrame, target_col: str = None) -> dict:
        """Prepare sequences for transformer model"""
        from sklearn.preprocessing import RobustScaler
        
        # Use only selected features
        feature_cols = [f for f in self.selected_features if f in df.columns]
        
        # Clean data
        df_clean = df[feature_cols + ([target_col] if target_col else [])].dropna()
        
        if len(df_clean) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + self.prediction_horizon}, got {len(df_clean)}")
        
        X, y = [], []
        
        for i in range(len(df_clean) - self.sequence_length - self.prediction_horizon + 1):
            X.append(df_clean[feature_cols].iloc[i:i+self.sequence_length].values)
            if target_col:
                y.append(df_clean[target_col].iloc[i+self.sequence_length+self.prediction_horizon-1])
        
        X = np.array(X)
        y = np.array(y) if target_col else None
        
        # Scale features
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
        
        X_scaled = X_scaled.reshape(X.shape)
        
        # Scale target
        if y is not None:
            if self.target_scaler is None:
                self.target_scaler = RobustScaler()
                y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        else:
            y_scaled = None
        
        return {
            'X': X_scaled,
            'y': y_scaled,
            'feature_names': feature_cols,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler
        }
    
    def _save_processed_features(self, train_sequences: dict, test_sequences: dict):
        """Save processed features to disk"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "processed_features.pkl", "wb") as f:
            pickle.dump({
                'train': train_sequences,
                'test': test_sequences,
                'selected_features': self.selected_features
            }, f)
        
        logger.info("✅ Processed features saved")
    
    def load_processed_features(self) -> Tuple[dict, dict]:
        """Load previously processed features"""
        with open("output/processed_features.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.selected_features = data['selected_features']
        logger.info("✅ Processed features loaded")
        
        return data['train'], data['test']
