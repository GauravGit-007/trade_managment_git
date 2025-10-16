# Refined AI Trading Signals

This folder contains the refined AI system that generates daily trading signals for 8 symbols using multiple data sources with equal weight.

## Overview

The Refined AI system processes:

- **8 Trading Symbols** (excluding Bitcoin & Ethereum)
- **3 Data Sources** with equal high weight:
  - Historical data (1-hour OHLC/trade data)
  - News articles (matched to symbols)
  - LSTM predictions (up to 12-hour predictions)

## Files

### `main_processor.py`

Main script that generates daily AI signals:

- Reads data from all three sources
- Combines analysis using Azure OpenAI GPT-4
- Outputs single refined decision per symbol per day
- Saves signals to `Smart_AI_decisions` database table

### `accuracy_checker.py`

Accuracy validation script:

- Compares AI decisions with actual market data
- Calculates accuracy metrics (MAE, MAPE, direction accuracy)
- Validates signal correctness (BUY/SELL/HOLD)
- Reports data availability and decision validity

### `requirements.txt`

Python dependencies for the refined AI system.

## Usage

### Generate Daily Signals

```bash
cd ai_signals/refined_AI
python main_processor.py
```

### Check Accuracy

```bash
cd ai_signals/refined_AI
python accuracy_checker.py
```

## Database Schema

### Smart_AI_decisions Table

- `id`: Unique identifier
- `symbol`: Trading symbol
- `decision_timestamp`: When decision was made
- `signal`: BUY/HOLD/SELL recommendation
- `confidence_score`: AI confidence (0.0-1.0)
- `predicted_price`: AI predicted price
- `current_price`: Price at decision time
- `reasoning`: AI explanation
- `data_sources_used`: Sources analyzed
- `created_at`: Record creation timestamp

## Symbols

The system processes these 8 symbols:

1. `/ES:XCME{=h}` - S&P 500 E-mini
2. `/NQ:XCME{=h}` - Nasdaq-100 E-mini
3. `/MES:XCME{=h}` - Micro S&P 500 E-mini
4. `/MNQ:XCME{=h}` - Micro Nasdaq-100 E-mini
5. `/RTY:XCME{=h}` - Russell 2000 E-mini
6. `/QM:XNYM{=h}` - Crude Oil E-mini
7. `/QG:XNYM{=h}` - Natural Gas E-mini
8. `/MCL:XNYM{=h}` - Micro Crude Oil

## Features

- **Daily Signal Generation**: One signal per symbol per day
- **Multi-Source Analysis**: Equal weight to historical, news, and LSTM data
- **Data Freshness Validation**: Checks if data sources are within 1-2 hours
- **Interactive Mode**: Prompts user to continue with outdated data
- **Non-Interactive Mode**: Automatically skips symbols with outdated data
- **Real-time Accuracy Tracking**: Compares predictions with actual outcomes
- **12-Hour Validity Window**: Signals valid for 12 hours after generation
- **Comprehensive Metrics**: MAE, MAPE, direction accuracy, signal accuracy

## Data Freshness Validation

The system includes a data freshness validation feature that ensures all data sources are current:

### Freshness Requirements

- **Historical Data**: Must be within 2 hours
- **News Data**: Must be within 2 hours
- **LSTM Predictions**: Must be within 2 hours

### Interactive Mode

When data is outdated, the system will:

1. Display detailed freshness status for each data source
2. Show how many hours old each data source is
3. Prompt user: "Do you want to continue with this outdated data?"
4. Allow user to choose: Continue (y) or Skip (n)

### Non-Interactive Mode

When using `--non-interactive` flag:

- Automatically skips symbols with outdated data
- No user prompts - suitable for automated runs
- Logs which symbols were skipped and why

## Output Files

- `daily_signals.json`: Latest generated signals
- `accuracy_report.json`: Detailed accuracy analysis results

# Generate daily signals (interactive mode)

python run_refined_ai.py generate

# Generate daily signals (non-interactive mode)

python run_refined_ai.py generate --non-interactive

# Check accuracy

python run_refined_ai.py check

# Run system tests

python run_refined_ai.py test

# Run complete workflow (interactive)

python run_refined_ai.py all

# Run complete workflow (non-interactive)

python run_refined_ai.py all --non-interactive

methos to test the ai signals

# Method 1: Direct processor

cd ai_signals/refined_AI/code
python main_processor.py

# Method 2: Using launcher (recommended)

cd ai_signals/refined_AI/code
python run_refined_ai.py generate

# Method 3: Non-interactive mode

python run_refined_ai.py generate --non-interactive
