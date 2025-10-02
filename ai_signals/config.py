# AI Signals Configuration
# This module contains all configuration settings for the AI signals system

import os
from datetime import datetime

class AIConfig:
    """Configuration class for AI signals system"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI = {
        'api_key': "71b66107a84e489ea700ef4188d29947",
        'azure_endpoint': "https://vastai-openai-swedencentral.openai.azure.com/",
        'api_version': "2024-02-15-preview",
        'deployment_name': "az-deployment",
        'model': "gpt4o",
        'temperature': 0.2,
        'max_tokens': 1000
    }
    
    # Trading Symbols
    TRADING_SYMBOLS = [
        "/ES:XCME{=h}",      # S&P 500 E-mini
        "/NQ:XCME{=h}",      # Nasdaq-100 E-mini
        "/MES:XCME{=h}",     # Micro S&P 500 E-mini
        "/MNQ:XCME{=h}",     # Micro Nasdaq-100 E-mini
        "/RTY:XCME{=h}",     # Russell 2000 E-mini
        "/QM:XNYM{=h}",      # Crude Oil E-mini
        "/QG:XNYM{=h}",      # Natural Gas E-mini
        "/MCL:XNYM{=h}",     # Micro Crude Oil
        "BTC/USD:CXTALP{=h}", # Bitcoin
        "ETH/USD:CXTALP{=h}"  # Ethereum
    ]
    
    # Action Mapping
    ACTION_MAPPING = {
        "STRONG_SELL": 0,
        "SELL": 1,
        "HOLD": 2,
        "BUY": 3,
        "STRONG_BUY": 4
    }
    
    # Technical Analysis Settings
    TECHNICAL_ANALYSIS = {
        'sma_periods': [5, 20],
        'rsi_period': 14,
        'atr_period': 14,
        'volume_sma_period': 5,
        'lookback_hours': 24
    }
    
    # News Analysis Settings
    NEWS_ANALYSIS = {
        'lookback_hours': 24,
        'max_articles_per_symbol': 5,
        'sentiment_threshold': 0.1,  # Minimum sentiment change to consider
        'confidence_threshold': 0.6  # Minimum confidence for trading
    }
    
    # Evaluation Settings
    EVALUATION = {
        'price_change_threshold': 0.001,  # 0.1% threshold for directional accuracy
        'evaluation_hours': 4,  # Hours to look forward for evaluation
        'min_accuracy_target': 70.0,  # Target accuracy percentage
        'min_win_rate_target': 60.0   # Target win rate percentage
    }
    
    # Monitoring Settings
    MONITORING = {
        'check_interval_minutes': 30,
        'alert_thresholds': {
            'min_accuracy': 60.0,
            'max_confidence_drop': 0.1,
            'max_consecutive_losses': 5,
            'min_win_rate': 50.0
        },
        'performance_history_size': 100
    }
    
    # Database Settings
    DATABASE = {
        'ai_decisions_table': 'ai_decisions',
        'evaluation_table': 'ai_evaluations',
        'monitoring_table': 'ai_monitoring'
    }
    
    # File Paths
    PATHS = {
        'output_dir': os.path.dirname(__file__),
        'signals_output': 'ai_signals_output.json',
        'evaluation_output': 'ai_evaluation_results.json',
        'monitoring_output': 'ai_monitoring_report.json',
        'log_file': 'ai_signals.log'
    }
    
    # Risk Management
    RISK_MANAGEMENT = {
        'max_position_size': 1.0,  # Maximum position size
        'stop_loss_percentage': 0.02,  # 2% stop loss
        'take_profit_percentage': 0.04,  # 4% take profit
        'max_daily_trades': 10,  # Maximum trades per day per symbol
        'min_confidence_for_trade': 0.7  # Minimum confidence to execute trade
    }
    
    # Performance Targets
    PERFORMANCE_TARGETS = {
        'min_accuracy': 70.0,
        'min_win_rate': 60.0,
        'max_drawdown': 10.0,
        'min_sharpe_ratio': 1.0,
        'max_consecutive_losses': 5
    }
    
    @classmethod
    def get_symbol_display_name(cls, symbol):
        """Get display name for symbol"""
        symbol_map = {
            "/ES:XCME{=h}": "S&P 500 E-mini",
            "/NQ:XCME{=h}": "Nasdaq-100 E-mini", 
            "/MES:XCME{=h}": "Micro S&P 500",
            "/MNQ:XCME{=h}": "Micro Nasdaq-100",
            "/RTY:XCME{=h}": "Russell 2000",
            "/QM:XNYM{=h}": "Crude Oil",
            "/QG:XNYM{=h}": "Natural Gas",
            "/MCL:XNYM{=h}": "Micro Crude Oil",
            "BTC/USD:CXTALP{=h}": "Bitcoin",
            "ETH/USD:CXTALP{=h}": "Ethereum"
        }
        return symbol_map.get(symbol, symbol)
    
    @classmethod
    def get_action_display_name(cls, action_code):
        """Get display name for action code"""
        reverse_mapping = {v: k for k, v in cls.ACTION_MAPPING.items()}
        return reverse_mapping.get(action_code, "UNKNOWN")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check required Azure OpenAI settings
        if not cls.AZURE_OPENAI['api_key']:
            errors.append("Azure OpenAI API key is required")
        
        if not cls.AZURE_OPENAI['azure_endpoint']:
            errors.append("Azure OpenAI endpoint is required")
        
        # Check trading symbols
        if not cls.TRADING_SYMBOLS:
            errors.append("At least one trading symbol is required")
        
        # Check performance targets
        if cls.PERFORMANCE_TARGETS['min_accuracy'] < 50:
            errors.append("Minimum accuracy target should be at least 50%")
        
        if cls.PERFORMANCE_TARGETS['min_win_rate'] < 40:
            errors.append("Minimum win rate target should be at least 40%")
        
        return errors
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("🔧 AI SIGNALS CONFIGURATION")
        print("=" * 40)
        print(f"📊 Trading Symbols: {len(cls.TRADING_SYMBOLS)}")
        print(f"🎯 Target Accuracy: {cls.PERFORMANCE_TARGETS['min_accuracy']}%")
        print(f"🏆 Target Win Rate: {cls.PERFORMANCE_TARGETS['min_win_rate']}%")
        print(f"⏰ Check Interval: {cls.MONITORING['check_interval_minutes']} minutes")
        print(f"📈 Lookback Hours: {cls.TECHNICAL_ANALYSIS['lookback_hours']}")
        print(f"🔍 News Lookback: {cls.NEWS_ANALYSIS['lookback_hours']} hours")
        print("=" * 40)

if __name__ == "__main__":
    # Validate and print configuration
    config = AIConfig()
    errors = config.validate_config()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid")
        config.print_config_summary()
