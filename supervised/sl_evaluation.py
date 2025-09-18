"""
Comprehensive evaluation tools for supervised learning models.
Provides backtesting, performance metrics, and comparison with RL models.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase
from supervised.infer_supervised import SupervisedInference
from rl.env import TradingEnv, EnvConfig


class SLEvaluator:
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sl_evaluation")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_model(self, model_path: str) -> Tuple[SupervisedInference, Dict]:
        """Load a trained SL model and its metadata."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        metadata_path = f"{model_path}.meta.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        metadata = joblib.load(metadata_path)
        model_type = metadata.get('model_type', 'lightgbm')
        
        inference = SupervisedInference(model_path, model_type, metadata.get('feature_cols'))
        
        return inference, metadata
    
    def backtest_model(self, model_path: str, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """Backtest a model on historical data."""
        inference, metadata = self.load_model(model_path)
        
        # Load historical data
        conn, cursor = TradeDatabase.sql_connect()
        
        # Get data for the symbol - handle the {=h} format properly
        db_symbol = symbol if "{=" in symbol else f"{symbol}{{=h}}"
        
        query = """
        SELECT timestamp, open, high, low, close, volume,
               rsi_14, ema_21, ema_50, MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9,
               BBL_20_2_0, BBM_20_2_0, BBU_20_2_0, BBB_20_2_0, BBP_20_2_0,
               atr_14, STOCHk_14_3_3, STOCHd_14_3_3
        FROM historical_data_1h
        WHERE symbol = ?
        ORDER BY timestamp
        """
        
        if start_date:
            query += " AND timestamp >= ?"
        if end_date:
            query += " AND timestamp <= ?"
        
        params = [db_symbol]
        if start_date:
            params.append(start_date)
        if end_date:
            params.append(end_date)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        TradeDatabase.close_connection(conn)
        
        if not rows:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Convert to DataFrame
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'rsi_14', 'ema_21', 'ema_50', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                'BBL_20_2_0', 'BBM_20_2_0', 'BBU_20_2_0', 'BBB_20_2_0', 'BBP_20_2_0',
                'atr_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
        
        df = pd.DataFrame(rows, columns=cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Simulate trading
        results = self._simulate_trading(df, inference, metadata)
        
        return results
    
    def _simulate_trading(self, df: pd.DataFrame, inference: SupervisedInference, metadata: Dict) -> Dict:
        """Simulate trading with the model."""
        window = metadata.get('window', 32)
        horizon = metadata.get('horizon', 15)
        feature_cols = metadata['feature_cols']
        
        # Initialize tracking variables
        position = 0.0
        cash = 10000.0  # Starting cash
        portfolio_value = []
        trades = []
        predictions = []
        
        # Transaction cost
        transaction_cost = 0.0005
        
        for i in range(window, len(df) - horizon):
            # Get current window of data
            window_data = df.iloc[i-window:i]
            
            # Prepare features
            features = window_data[feature_cols].values.flatten()
            
            # Get prediction
            try:
                action = inference.predict_action(features)
                predictions.append(action)
            except Exception as e:
                print(f"Prediction error at index {i}: {e}")
                predictions.append(0)  # Hold
                continue
            
            # Execute trade
            current_price = df.iloc[i]['close']
            next_price = df.iloc[i + horizon]['close']
            
            # Calculate position change
            if action == 4:  # Buy
                position_change = 1.0
            elif action == 0:  # Sell
                position_change = -1.0
            else:  # Hold
                position_change = 0.0
            
            # Update position and cash
            if position_change != 0:
                cost = abs(position_change) * current_price * transaction_cost
                cash -= cost
                position += position_change
                
                trades.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'action': action,
                    'price': current_price,
                    'position_change': position_change,
                    'cost': cost
                })
            
            # Calculate portfolio value
            portfolio_val = cash + position * current_price
            portfolio_value.append({
                'timestamp': df.iloc[i]['timestamp'],
                'portfolio_value': portfolio_val,
                'position': position,
                'cash': cash,
                'price': current_price
            })
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        if len(portfolio_df) > 0:
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            
            metrics = {
                'total_return': (portfolio_df['portfolio_value'].iloc[-1] - 10000) / 10000,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio_df['portfolio_value']),
                'num_trades': len(trades),
                'win_rate': self._calculate_win_rate(trades, df),
                'avg_trade_return': self._calculate_avg_trade_return(trades, df),
                'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
                'final_position': position,
                'final_cash': cash
            }
        else:
            metrics = {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'final_portfolio_value': 10000,
                'final_position': 0,
                'final_cash': 10000
            }
        
        return {
            'metrics': metrics,
            'trades': trades,
            'portfolio_history': portfolio_value,
            'predictions': predictions
        }
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def _calculate_win_rate(self, trades: List[Dict], df: pd.DataFrame) -> float:
        """Calculate win rate of trades."""
        if not trades:
            return 0.0
        
        wins = 0
        for trade in trades:
            # Find the price at horizon
            trade_time = trade['timestamp']
            horizon_time = trade_time + timedelta(hours=15)  # Assuming 15-hour horizon
            
            # Find closest price after horizon
            future_prices = df[df['timestamp'] > horizon_time]
            if len(future_prices) > 0:
                future_price = future_prices.iloc[0]['close']
                trade_return = (future_price - trade['price']) / trade['price']
                
                if trade['position_change'] > 0 and trade_return > 0:  # Buy and price went up
                    wins += 1
                elif trade['position_change'] < 0 and trade_return < 0:  # Sell and price went down
                    wins += 1
        
        return wins / len(trades) if trades else 0.0
    
    def _calculate_avg_trade_return(self, trades: List[Dict], df: pd.DataFrame) -> float:
        """Calculate average return per trade."""
        if not trades:
            return 0.0
        
        total_return = 0.0
        for trade in trades:
            trade_time = trade['timestamp']
            horizon_time = trade_time + timedelta(hours=15)
            
            future_prices = df[df['timestamp'] > horizon_time]
            if len(future_prices) > 0:
                future_price = future_prices.iloc[0]['close']
                trade_return = (future_price - trade['price']) / trade['price']
                
                if trade['position_change'] > 0:  # Buy
                    total_return += trade_return
                elif trade['position_change'] < 0:  # Sell
                    total_return -= trade_return
        
        return total_return / len(trades)
    
    def compare_models(self, model_paths: List[str], symbol: str) -> Dict:
        """Compare multiple models on the same data."""
        results = {}
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).split('.')[0]
            try:
                backtest_results = self.backtest_model(model_path, symbol)
                results[model_name] = backtest_results['metrics']
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.dropna()
        
        # Save comparison
        comparison_path = os.path.join(self.results_dir, f"model_comparison_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        comparison_df.to_csv(comparison_path)
        
        return {
            'comparison_df': comparison_df,
            'comparison_path': comparison_path
        }
    
    def generate_report(self, model_path: str, symbol: str) -> str:
        """Generate a comprehensive evaluation report."""
        inference, metadata = self.load_model(model_path)
        
        # Run backtest
        backtest_results = self.backtest_model(model_path, symbol)
        
        # Generate report
        report = f"""
# Supervised Learning Model Evaluation Report

## Model Information
- **Model Path**: {model_path}
- **Model Type**: {metadata.get('model_type', 'unknown')}
- **Symbol**: {symbol}
- **Training Date**: {metadata.get('training_date', 'unknown')}
- **Horizon**: {metadata.get('horizon', 'unknown')} minutes

## Performance Metrics
- **Total Return**: {backtest_results['metrics']['total_return']:.2%}
- **Sharpe Ratio**: {backtest_results['metrics']['sharpe_ratio']:.3f}
- **Max Drawdown**: {backtest_results['metrics']['max_drawdown']:.2%}
- **Number of Trades**: {backtest_results['metrics']['num_trades']}
- **Win Rate**: {backtest_results['metrics']['win_rate']:.2%}
- **Average Trade Return**: {backtest_results['metrics']['avg_trade_return']:.2%}
- **Final Portfolio Value**: ${backtest_results['metrics']['final_portfolio_value']:.2f}

## Trading Summary
- **Final Position**: {backtest_results['metrics']['final_position']:.2f}
- **Final Cash**: ${backtest_results['metrics']['final_cash']:.2f}

## Model Configuration
```json
{json.dumps(metadata.get('config', {}), indent=2)}
```

## Feature Columns
{', '.join(metadata.get('feature_cols', []))}
"""
        
        # Save report
        report_path = os.path.join(self.results_dir, f"evaluation_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate supervised learning models")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--symbol", required=True, help="Symbol to evaluate")
    parser.add_argument("--start_date", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--compare", nargs="+", help="Additional models to compare")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    evaluator = SLEvaluator()
    
    if args.compare:
        # Compare multiple models
        all_models = [args.model] + args.compare
        results = evaluator.compare_models(all_models, args.symbol)
        print("Model Comparison Results:")
        print(results['comparison_df'])
        print(f"Comparison saved to: {results['comparison_path']}")
    else:
        # Single model evaluation
        results = evaluator.backtest_model(args.model, args.symbol, args.start_date, args.end_date)
        print("Backtest Results:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value}")
    
    if args.report:
        report_path = evaluator.generate_report(args.model, args.symbol)
        print(f"Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()

