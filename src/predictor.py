"""
Prediction generation module
"""

import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from .model_trainer import VolatilityTransformer, VolatilityDataset

logger = logging.getLogger(__name__)


class Predictor:
    """Handles model prediction and submission generation"""
    
    def __init__(self, model_path: str = "models/best_model.pth", batch_size: int = 64):
        """
        Initialize Predictor
        
        Args:
            model_path: Path to trained model
            batch_size: Batch size for prediction
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def predict(self, test_sequences: Dict) -> np.ndarray:
        """
        Generate predictions for test data
        
        Args:
            test_sequences: Dictionary containing test sequences
            
        Returns:
            Array of predictions
        """
        X_test = test_sequences['X']
        target_scaler = test_sequences['target_scaler']
        
        # Load model if not already loaded
        if self.model is None:
            self._load_model(X_test.shape[-1])
        
        logger.info("Generating predictions...")
        
        self.model.eval()
        predictions_scaled = []
        
        test_dataset = VolatilityDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_X in test_loader:
                if isinstance(batch_X, tuple):
                    batch_X = batch_X[0]
                
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X).cpu().numpy()
                predictions_scaled.extend(batch_pred)
        
        # Convert to numpy and inverse transform
        predictions_scaled = np.array(predictions_scaled)
        predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
        
        return predictions
    
    def _load_model(self, input_dim: int):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = VolatilityTransformer(input_dim=input_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        
        logger.info(f"✅ Model loaded from {self.model_path}")
    
    def save_submission(self, predictions: np.ndarray, output_path: str):
        """
        Save predictions in submission format
        
        Args:
            predictions: Array of predictions
            output_path: Path to save submission file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        submission_df = pd.DataFrame({
            'timestamp': range(1, len(predictions) + 1),
            'labels': predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        logger.info(f"✅ Submission saved to {output_path}")
        
        return submission_df
