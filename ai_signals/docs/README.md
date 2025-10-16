# AI Signals Trading System

A comprehensive AI-powered trading signal generation system that uses Azure OpenAI to analyze market data, news sentiment, and technical indicators to generate high-accuracy trading signals.

## 🎯 Overview

This system is designed to achieve **>70% accuracy** in trading signal generation by combining:

- Real-time market data analysis
- News sentiment analysis using Azure OpenAI
- Technical indicator calculations
- Comprehensive risk management
- Real-time performance monitoring

## 🏗️ System Architecture

```
ai_signals/
├── ai_decision_engine.py    # Core AI signal generation
├── ai_evaluation.py         # Performance evaluation system
├── ai_monitoring.py         # Real-time monitoring & alerts
├── ai_workflow.py          # Orchestration workflow
├── config.py               # Configuration settings
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install openai pandas numpy python-dotenv
```

### 2. Database Setup

Make sure your database has the required tables:

- `historical_data_1h` - Hourly price data
- `news_articles` - News articles
- `sentiment_analysis` - News sentiment scores
- `positions` - Current positions

### 3. Run AI Signals

#### Generate Signals Only

```bash
python ai_workflow.py --mode signals
```

#### Full Workflow (Signals + Evaluation + Monitoring)

```bash
python ai_workflow.py --mode full
```

#### Evaluate Recent Performance

```bash
python ai_workflow.py --mode evaluate --hours 24
```

#### Monitor System Performance

```bash
python ai_workflow.py --mode monitor --hours 6
```

#### Continuous Monitoring

```bash
python ai_workflow.py --mode continuous --interval 30
```

## 📊 Trading Symbols

The system currently supports these symbols:

- **Equity Indices**: /ES:XCME, /NQ:XCME, /MES:XCME, /MNQ:XCME, /RTY:XCME
- **Commodities**: /QM:XNYM, /QG:XNYM, /MCL:XNYM
- **Cryptocurrencies**: BTC/USD:CXTALP, ETH/USD:CXTALP

## 🤖 AI Decision Process

### 1. Data Collection

- **Price Data**: Current and historical OHLCV data
- **Technical Indicators**: SMA, RSI, ATR, Volume analysis
- **News Sentiment**: Recent news articles with sentiment scores
- **Market Context**: Current positions and market conditions

### 2. AI Analysis

The system uses Azure OpenAI (GPT-4) to analyze:

- Technical indicator alignment
- News sentiment impact
- Market volatility and trends
- Risk-reward ratios
- Current position management

### 3. Signal Generation

AI generates signals with:

- **Action**: STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY
- **Confidence**: 0.0 to 1.0
- **Reasoning**: Detailed explanation
- **Price Targets**: Entry, stop-loss, take-profit levels
- **Risk Metrics**: Risk-reward ratio, time horizon

## 📈 Performance Monitoring

### Real-time Metrics

- **Directional Accuracy**: Percentage of correct predictions
- **Win Rate**: Percentage of profitable trades
- **Average Confidence**: AI confidence levels
- **P&L Tracking**: Profit/loss per trade
- **Symbol Performance**: Breakdown by trading symbol

### Alert System

- Accuracy below threshold (< 60%)
- Low confidence levels (< 0.5)
- Consecutive losing trades (> 5)
- System performance degradation

## ⚙️ Configuration

Key configuration settings in `config.py`:

```python
# Performance Targets
PERFORMANCE_TARGETS = {
    'min_accuracy': 70.0,      # Target accuracy %
    'min_win_rate': 60.0,      # Target win rate %
    'max_drawdown': 10.0,      # Maximum drawdown %
}

# Risk Management
RISK_MANAGEMENT = {
    'max_position_size': 1.0,
    'stop_loss_percentage': 0.02,  # 2%
    'take_profit_percentage': 0.04,  # 4%
    'min_confidence_for_trade': 0.7
}
```

## 📋 Output Files

The system generates several output files:

- `ai_signals_output.json` - Latest AI signals for all symbols
- `ai_evaluation_results.json` - Performance evaluation results
- `ai_monitoring_report.json` - Real-time monitoring report

## 🔧 Integration with Existing System

This AI signals system integrates with your existing trade management infrastructure:

1. **Database Integration**: Uses existing `TradeDatabase` class
2. **Data Sources**: Leverages existing historical data and news services
3. **Action Mapping**: Compatible with existing RL/SL action codes
4. **Position Management**: Integrates with existing position tracking

## 📊 Expected Performance

Based on the AI analysis approach, the system targets:

- **Accuracy**: >70% directional accuracy
- **Win Rate**: >60% profitable trades
- **Confidence**: High-confidence signals (>0.7)
- **Risk Management**: Controlled drawdowns and position sizing

## 🚨 Monitoring & Alerts

The system provides comprehensive monitoring:

### Performance Alerts

- Accuracy drops below 60%
- Consecutive losses exceed 5 trades
- Confidence levels consistently low
- System errors or data issues

### Real-time Dashboard

- Live accuracy tracking
- Symbol performance breakdown
- Confidence level monitoring
- P&L tracking

## 🔄 Workflow Integration

### Manual Workflow

```bash
# 1. Generate signals
python ai_workflow.py --mode signals

# 2. Evaluate performance
python ai_workflow.py --mode evaluate

# 3. Monitor system
python ai_workflow.py --mode monitor
```

### Automated Workflow

```bash
# Run full workflow every 30 minutes
python ai_signals\ai_workflow.py --mode continuous --interval 30
```

## 🛠️ Troubleshooting

### Common Issues

1. **No signals generated**: Check database connectivity and data availability
2. **Low accuracy**: Review news sentiment data and technical indicators
3. **High confidence but poor performance**: Adjust AI prompt or evaluation criteria
4. **Database errors**: Ensure all required tables exist and are populated

### Debug Mode

Enable detailed logging by setting environment variable:

```bash
export AI_DEBUG=1
```

## 📞 Support

For issues or questions:

1. Check the monitoring reports for system status
2. Review evaluation results for performance insights
3. Examine individual signal reasoning for decision quality
4. Monitor database connectivity and data freshness

## 🎯 Success Metrics

The system is considered successful when:

- ✅ Accuracy consistently >70%
- ✅ Win rate >60%
- ✅ Low false positive rate
- ✅ Stable confidence levels
- ✅ Effective risk management

---

**Note**: This system uses real market data and AI analysis. Always validate signals before making actual trades and consider your risk tolerance and investment objectives.
