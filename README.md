# ETH Implied Volatility Prediction using Multi-head Attention Transformer

Deep attention-based time series model for cryptocurrency implied volatility prediction using order book imbalance and cross-asset correlation features.

## Project Overview

This project implements a tuned Transformer for cryptocurrency implied volatility forecasting, specifically designed for quantitative trading applications. The model processes 1-second resolution order book data to predict implied volatility 10 seconds into the future, achieving competitive performance on financial time series prediction.

### Key Features

- **Transformer Architecture**: Custom implementation optimized for financial time series
- **High-Frequency Data Processing**: Handles 1-second resolution order book snapshots
- **Comprehensive Feature Engineering**: 50+ engineered features from market microstructure
- **Real-time Ready**: Designed for low-latency trading environments
- **Production Pipeline**: Complete ML pipeline from data preprocessing to model deployment

## Performance Metrics

- **Evaluation Metric**: Pearson Correlation Score
- **Model Architecture**: Transformer Encoder with 4 layers, 8 attention heads
- **Sequence Length**: 60 seconds (1-minute lookback)
- **Prediction Horizon**: 10 seconds ahead
- **Training Data**: 50,000 samples per cryptocurrency

## Architecture

### Model Components

1. **Input Projection Layer**: Projects features to model dimension (128)
2. **Positional Encoding**: Time-aware positional embeddings
3. **Transformer Encoder**: 4-layer encoder with multi-head attention
4. **Global Average Pooling**: Temporal aggregation
5. **Feed-Forward Network**: 3-layer MLP for final prediction

### Feature Engineering Pipeline

- **Order Book Features**: Bid-ask spread, volume imbalance, weighted prices
- **Technical Indicators**: Moving averages, momentum, volatility measures
- **Lagged Features**: Historical price and volume patterns
- **Cross-Asset Signals**: Multi-cryptocurrency correlations

## Installation & Setup

### Environment Setup

1. **Clone the repository**:

```bash
git clone https://github.com/saomyaraj/VolatilityNet.git
cd VolatilityNet
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Dataset

### Data Source

You can obtain the cryptocurrency implied volatility forecasting dataset from:

- **Primary Source**: [Kaggle Competition - Crypto Implied Volatility Forecasting](https://www.kaggle.com/competitions/gq-implied-volatility-forecasting)

The dataset contains high-frequency order book data for 7 cryptocurrencies:

- **Assets**: ETH, BTC, DOGE, DOT, LINK, SHIB, SOL
- **Frequency**: 1-second resolution
- **Features**: Order book snapshots with 5 bid/ask levels
- **Target**: Implied volatility labels (training data only)
- **Time Period**: Multiple months of continuous trading data

## Quick Start

### 1. Run the Complete Pipeline

```bash
jupyter notebook eth_volatility_prediction.ipynb
```

Execute all cells sequentially to:

- Load and explore the data
- Engineer features
- Train the Transformer model
- Generate predictions
- Analyze results

### 2. Model Training

The notebook automatically:

- Splits data using time series validation (80/20)
- Trains for 30 epochs with early stopping
- Saves best model weights to `best_model.pth`
- Generates training/validation loss plots

### 3. Generate Predictions

Final predictions are saved to `transformer_submission.csv` in the required format:

```csv
timestamp,labels
1,0.00012345
2,0.00023456
...
```

## Trading Strategy Framework

### Volatility Regime Detection

- **Low Volatility** (< 25th percentile): Sell volatility strategies
- **High Volatility** (> 75th percentile): Buy volatility strategies
- **Medium Volatility**: Delta-neutral approaches

### Risk Management

- **Position Sizing**: Based on prediction confidence
- **Stop Losses**: 2-3x expected volatility moves
- **Maximum Exposure**: 5% portfolio allocation per position

### Real-time Implementation

- **Latency Target**: < 5ms total system latency
- **Memory Usage**: < 500MB runtime memory
- **Throughput**: 1000+ predictions per second

## Production Deployment

### Model Serving

```python
# Load trained model for inference
model = VolatilityTransformer(input_dim=45)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Real-time prediction pipeline
def predict_volatility(order_book_data):
    features = engineer_features(order_book_data)
    sequence = prepare_sequence(features)
    with torch.no_grad():
        prediction = model(sequence)
    return prediction.item()
```

### Monitoring & Maintenance

- **Model Drift Detection**: Statistical tests on prediction distributions
- **Performance Monitoring**: Real-time correlation tracking
- **Retraining Schedule**: Weekly model updates with new data
- **A/B Testing**: Gradual rollout of model improvements

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Submit pull request with detailed description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
