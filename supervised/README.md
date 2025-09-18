# Supervised Learning Implementation for Trade Management

This folder contains a comprehensive supervised learning implementation that mirrors the reinforcement learning functionality while providing an alternative approach to trading decisions.

## Overview

The supervised learning approach converts the trading problem into a classification task where the model learns to predict optimal actions (buy, hold, sell) based on historical market data and future returns.

## Architecture

### Core Components

1. **Data Pipeline**

   - `make_labels.py` - Creates classification labels from future returns
   - `dataset.py` - PyTorch dataset for time-series data
   - `sl_training_pipeline.py` - Comprehensive training pipeline

2. **Model Training**

   - `train_baseline.py` - LightGBM baseline trainer
   - `train_pytorch.py` - PyTorch CNN trainer
   - `sl_training_pipeline.py` - Unified training pipeline with cross-validation

3. **Inference & Deployment**

   - `infer_supervised.py` - Model inference wrapper
   - `sl_decision_loop.py` - Live trading decision loop
   - `sl_workflow.py` - Complete workflow management

4. **Evaluation & Monitoring**
   - `sl_evaluation.py` - Backtesting and performance evaluation
   - `sl_monitoring.py` - Real-time model monitoring and alerts

## Quick Start

### 1. Full Pipeline (Recommended)

```bash
# Run complete pipeline from data prep to deployment
python supervised/sl_workflow.py --action full_pipeline --symbols /ES:XCME /NQ:XCME
```

### 2. Step-by-Step Approach

#### Prepare Training Data

```bash
python supervised/sl_workflow.py --action prepare_data --symbols /ES:XCME /NQ:XCME
```

#### Train Models

```bash
# Train both LightGBM and PyTorch models
python supervised/sl_workflow.py --action train --symbols /ES:XCME /NQ:XCME

# Or train specific model type
python supervised/sl_training_pipeline.py --model_type lightgbm --symbols /ES:XCME
```

#### Evaluate Models

```bash
python supervised/sl_workflow.py --action evaluate --model_paths models/lightgbm_sl_all_*.txt models/pytorch_sl_all_*.pth
```

#### Deploy Best Model

```bash
python supervised/sl_workflow.py --action deploy --model models/lightgbm_sl_all_20240101_120000.txt
```

### 3. Manual Training (Legacy)

#### Create Labels

```bash
python supervised/make_labels.py --candles db/processed_candles_with_ta.csv --horizon 15 --out data/processed_with_labels_15m.parquet
```

#### Train Baseline Model

```bash
python supervised/train_baseline.py --candles db/processed_candles_with_ta.csv --labels data/processed_with_labels_15m.parquet --label_col label_15 --model_out models/lightgbm_baseline.txt
```

#### Test Inference

```bash
python supervised/infer_supervised.py --model models/lightgbm_baseline.txt --model_type lightgbm --state_json examples/state_example.json
```

## Configuration

### Config File (config.yaml)

```yaml
labels:
  horizon_minutes: 15 # Prediction horizon
  threshold: 0.001 # Return threshold for classification
  cost: 0.0005 # Transaction cost

dataset:
  window: 32 # Time window for features
  stride: 1 # Window stride
  test_size: 0.2 # Test set size
  validation_size: 0.1 # Validation set size

training:
  baseline:
    model_type: lightgbm
    num_boost_round: 200
    early_stopping_rounds: 20
    learning_rate: 0.05
    num_leaves: 31
  pytorch:
    epochs: 50
    batch_size: 64
    lr: 0.001
    patience: 10
```

## Model Types

### 1. LightGBM (Recommended)

- **Pros**: Fast training, good performance, interpretable
- **Use Case**: Baseline model, production deployment
- **Output**: Classification probabilities

### 2. PyTorch CNN

- **Pros**: Can capture complex patterns, flexible architecture
- **Use Case**: Research, complex feature interactions
- **Output**: Classification logits

## Features

The models use the same features as the RL environment:

- **Price Data**: OHLCV, technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Engineered Features**: Price ratios, volatility measures, momentum indicators
- **Sentiment Data**: News sentiment scores (if available)
- **LSTM Predictions**: Future price predictions (if available)

## Performance Metrics

### Trading Metrics

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Return**: Mean return per trade

### Model Metrics

- **Accuracy**: Classification accuracy
- **Precision/Recall**: Per-class performance
- **Confidence**: Prediction confidence scores

## Monitoring & Alerts

The monitoring system tracks:

- **Model Performance**: Real-time P&L tracking
- **Decision Quality**: Confidence scores and win rates
- **Risk Metrics**: Drawdown and loss streaks
- **Alerts**: Automated notifications for performance issues

### Start Monitoring

```bash
python supervised/sl_monitoring.py --model models/lightgbm_sl_all_*.txt --interval 5
```

## Integration with RL System

The SL implementation is designed to work alongside the RL system:

1. **Shared Environment**: Uses the same `TradingEnv` for state building
2. **Consistent Actions**: Maps to same action space as RL (0-4)
3. **Database Integration**: Logs to `sl_decisions` table
4. **Parallel Deployment**: Can run both RL and SL simultaneously

## File Structure

```
supervised/
├── README.md                    # This file
├── config.yaml                  # Configuration
├── dataset.py                   # PyTorch dataset
├── make_labels.py              # Label creation
├── train_baseline.py           # LightGBM training
├── train_pytorch.py            # PyTorch training
├── sl_training_pipeline.py     # Unified training
├── infer_supervised.py         # Model inference
├── sl_decision_loop.py         # Live trading loop
├── sl_evaluation.py            # Backtesting
├── sl_monitoring.py            # Real-time monitoring
├── sl_workflow.py              # Workflow management
├── examples/                   # Example files
└── tests/                      # Unit tests
```

## Best Practices

1. **Data Quality**: Ensure clean, properly formatted data
2. **Feature Engineering**: Experiment with different feature combinations
3. **Model Selection**: Use cross-validation for model comparison
4. **Risk Management**: Monitor drawdown and position sizes
5. **Regular Retraining**: Update models with fresh data
6. **A/B Testing**: Compare SL vs RL performance

## Troubleshooting

### Common Issues

1. **No Data**: Check database connection and data availability
2. **Training Errors**: Verify feature columns and data types
3. **Poor Performance**: Adjust hyperparameters or feature engineering
4. **Memory Issues**: Reduce batch size or window size

### Debug Mode

```bash
# Enable verbose logging
export SL_DEBUG=1
python supervised/sl_workflow.py --action full_pipeline
```

## Performance Comparison

The SL approach offers several advantages over RL:

- **Faster Training**: No environment simulation required
- **More Stable**: Deterministic training process
- **Interpretable**: Feature importance and decision explanations
- **Easier Debugging**: Clear input-output relationships

However, RL may be better for:

- **Dynamic Environments**: Adapting to changing market conditions
- **Long-term Planning**: Multi-step decision optimization
- **Exploration**: Discovering new trading strategies
