"""
Transformer model architecture and training module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class VolatilityTransformer(nn.Module):
    """Transformer model for volatility prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super(VolatilityTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, 1)
        
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)


class VolatilityDataset(Dataset):
    """Dataset class for volatility data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_path: str = "models/best_model.pth", 
                 batch_size: int = 32, learning_rate: float = 0.001,
                 num_epochs: int = 30):
        """
        Initialize ModelTrainer
        
        Args:
            model_path: Path to save/load model
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models directory
        self.model_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
    
    def train(self, train_sequences: Dict) -> Tuple[VolatilityTransformer, Dict]:
        """
        Train the transformer model
        
        Args:
            train_sequences: Dictionary containing training sequences
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        X_train = train_sequences['X']
        y_train = train_sequences['y']
        target_scaler = train_sequences['target_scaler']
        
        # Time series split
        split_idx = int(0.8 * len(X_train))
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val_split = X_train[split_idx:]
        y_val_split = y_train[split_idx:]
        
        logger.info(f"Training split: {X_train_split.shape}, Validation split: {X_val_split.shape}")
        
        # Create data loaders
        train_dataset = VolatilityDataset(X_train_split, y_train_split)
        val_dataset = VolatilityDataset(X_val_split, y_val_split)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X_train.shape[-1]
        model = VolatilityTransformer(input_dim=input_dim)
        model = model.to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training
        train_losses, val_losses = self._train_model(model, train_loader, val_loader)
        
        # Evaluation
        metrics = self._evaluate_model(model, X_val_split, y_val_split, target_scaler)
        
        logger.info(f"Training completed. Correlation: {metrics['correlation']:.4f}")
        
        return model, metrics
    
    def _train_model(self, model: VolatilityTransformer, train_loader: DataLoader, 
                    val_loader: DataLoader) -> Tuple[list, list]:
        """Train the model"""
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.model_path)
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch:3d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
    
    def _evaluate_model(self, model: VolatilityTransformer, X_val: np.ndarray, 
                       y_val: np.ndarray, target_scaler) -> Dict:
        """Evaluate model performance"""
        model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions_scaled = model(X_val_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        correlation, _ = pearsonr(y_true, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'predictions': predictions,
            'y_true': y_true
        }
