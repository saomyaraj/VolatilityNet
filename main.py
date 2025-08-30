"""
VolatilityNet: ETH Implied Volatility Prediction
Main execution script for the complete ML pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from src.utils import setup_logging, save_config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='VolatilityNet: ETH Implied Volatility Prediction')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['data', 'train', 'predict', 'full'],
                       help='Execution mode: data, train, predict, or full pipeline')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--output', type=str, default='output/submission.csv',
                       help='Output prediction file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting VolatilityNet Pipeline")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode in ['data', 'full']:
            # Data Loading and Feature Engineering
            logger.info("📊 Loading and processing data...")
            data_loader = DataLoader()
            train_data, test_data = data_loader.load_data()
            
            feature_engineer = FeatureEngineer()
            train_features, test_features = feature_engineer.create_features(train_data, test_data)
            logger.info("✅ Data processing completed")
        
        if args.mode in ['train', 'full']:
            # Model Training
            logger.info("🤖 Training model...")
            trainer = ModelTrainer(model_path=args.model_path)
            if args.mode == 'train':
                # Load pre-processed features if only training
                feature_engineer = FeatureEngineer()
                train_features, test_features = feature_engineer.load_processed_features()
            
            model, metrics = trainer.train(train_features)
            logger.info(f"✅ Model training completed. Correlation: {metrics['correlation']:.4f}")
        
        if args.mode in ['predict', 'full']:
            # Prediction Generation
            logger.info("🎯 Generating predictions...")
            predictor = Predictor(model_path=args.model_path)
            if args.mode == 'predict':
                # Load pre-processed features if only predicting
                feature_engineer = FeatureEngineer()
                train_features, test_features = feature_engineer.load_processed_features()
                
            predictions = predictor.predict(test_features)
            predictor.save_submission(predictions, args.output)
            logger.info(f"✅ Predictions saved to {args.output}")
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
