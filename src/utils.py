"""
Utility functions and helpers
"""

import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any


def setup_logging(level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('output/volatility_net.log')
        ]
    )
    
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device_info() -> Dict[str, Any]:
    """Get information about available computing devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    return info


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate model size and parameter count"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb
    }
