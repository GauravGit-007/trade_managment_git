# Refined AI Configuration
# Configuration settings for the refined AI trading system

class RefinedAIConfig:
    """Configuration class for Refined AI system"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI = {
        'api_key': "71b66107a84e489ea700ef4188d29947",
        'azure_endpoint': "https://vastai-openai-swedencentral.openai.azure.com/",
        'api_version': "2024-02-15-preview",
        'deployment_name': "az-deployment",
        'model': "gpt4o",
        'temperature': 0.3,
        'max_tokens': 1000
    }
    
    # Trading Symbols (8 symbols excluding Bitcoin & Ethereum)
    SYMBOLS = [
        "/ES:XCME{=h}",      # S&P 500 E-mini
        "/NQ:XCME{=h}",      # Nasdaq-100 E-mini
        "/MES:XCME{=h}",     # Micro S&P 500 E-mini
        "/MNQ:XCME{=h}",     # Micro Nasdaq-100 E-mini
        "/RTY:XCME{=h}",     # Russell 2000 E-mini
        "/QM:XNYM{=h}",      # Crude Oil E-mini
        "/QG:XNYM{=h}",      # Natural Gas E-mini
        "/MCL:XNYM{=h}"      # Micro Crude Oil
    ]
    
    # Data Processing Settings
    DATA_PROCESSING = {
        'historical_hours_back': 24,      # Hours of historical data to analyze
        'news_hours_back': 24,            # Hours of news data to analyze
        'lstm_hours_ahead': 12,           # Hours ahead for LSTM predictions
        'min_historical_data': 5,         # Minimum historical data points required
        'max_news_articles': 10           # Maximum news articles to analyze
    }
    
    # Technical Analysis Settings
    TECHNICAL_ANALYSIS = {
        'sma_short': 5,                   # Short-term SMA period
        'sma_long': 20,                   # Long-term SMA period
        'rsi_period': 14,                 # RSI calculation period
        'volume_sma_period': 5,           # Volume SMA period
        'price_change_threshold': 0.5     # Minimum price change to consider significant
    }
    
    # AI Decision Settings
    AI_DECISION = {
        'confidence_threshold': 0.6,      # Minimum confidence for trading
        'signal_validity_hours': 12,      # Hours signal remains valid
        'price_prediction_weight': 0.33,  # Weight for price prediction
        'news_sentiment_weight': 0.33,    # Weight for news sentiment
        'technical_weight': 0.34          # Weight for technical analysis
    }
    
    # Data Freshness Settings
    DATA_FRESHNESS = {
        'max_hours_old': 2,               # Maximum hours data can be old
        'interactive_mode': True,          # Default interactive mode
        'skip_outdated': False,           # Skip outdated data in non-interactive mode
        'warn_threshold': 1,              # Hours before warning about data age
        'critical_threshold': 4           # Hours before marking data as critical
    }
    
    # Accuracy Check Settings
    ACCURACY_CHECK = {
        'max_hours_back': 24,             # Maximum hours to look back for decisions
        'min_hours_data_required': 1,     # Minimum hours of actual data required
        'price_accuracy_threshold': 0.05, # 5% price accuracy threshold
        'signal_accuracy_threshold': 0.6  # 60% signal accuracy threshold
    }
    
    # Database Settings
    DATABASE = {
        'table_name': 'Smart_AI_decisions',
        'backup_enabled': True,
        'log_level': 'INFO'
    }
    
    # Output Settings
    OUTPUT = {
        'save_json': True,
        'json_filename': 'daily_signals.json',
        'accuracy_filename': 'accuracy_report.json',
        'log_to_console': True
    }
