# Refined AI Accuracy Checker
# Compares AI decisions with actual market data to calculate accuracy metrics

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
root_dir = os.path.dirname(grandparent_dir)
sys.path.append(root_dir)
from db.database import TradeDatabase

class RefinedAIAccuracyChecker:
    def __init__(self):
        """Initialize the Accuracy Checker"""
        self.db = TradeDatabase()
        self.symbols = [
            "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
            "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}"
        ]
        # Optional filters
        self.min_confidence: float | None = None
        self.signal_whitelist: set[str] | None = None  # e.g., {"BUY","SELL"}
        self.latest_per_symbol: bool = False
    
    def get_ai_decisions(self, symbol=None, hours_back=24):
        """Get AI decisions from Smart_AI_decisions table"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return []
        
        try:
            if symbol:
                query = """
                SELECT id, symbol, decision_timestamp, signal, confidence_score, 
                       predicted_price, current_price, reasoning, created_at
                FROM Smart_AI_decisions
                WHERE symbol = ?
                AND datetime(decision_timestamp) >= datetime('now', '-{} hours')
                ORDER BY decision_timestamp DESC
                """.format(hours_back)
                cursor.execute(query, (symbol,))
            else:
                query = """
                SELECT id, symbol, decision_timestamp, signal, confidence_score, 
                       predicted_price, current_price, reasoning, created_at
                FROM Smart_AI_decisions
                WHERE datetime(decision_timestamp) >= datetime('now', '-{} hours')
                ORDER BY symbol, decision_timestamp DESC
                """.format(hours_back)
                cursor.execute(query)
            
            rows = cursor.fetchall()
            decisions = []
            
            for row in rows:
                decisions.append({
                    'id': row[0],
                    'symbol': row[1],
                    'decision_timestamp': row[2],
                    'signal': row[3],
                    'confidence_score': row[4],
                    'predicted_price': row[5],
                    'current_price': row[6],
                    'reasoning': row[7],
                    'created_at': row[8]
                })
            
            return decisions
            
        except Exception as e:
            print(f"Error fetching AI decisions: {e}")
            return []
        finally:
            self.db.close_connection(conn)
    
    def get_actual_prices_after_decision(self, symbol, decision_timestamp, hours_ahead=12):
        """Get actual prices after a decision timestamp"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return []
        
        try:
            query = """
            SELECT close, timestamp
            FROM historical_data_1h
            WHERE symbol = ?
            AND datetime(timestamp) > datetime(?)
            AND datetime(timestamp) <= datetime(?, '+{} hours')
            ORDER BY timestamp ASC
            """.format(hours_ahead)
            
            cursor.execute(query, (symbol, decision_timestamp, decision_timestamp))
            rows = cursor.fetchall()
            
            actual_prices = []
            for row in rows:
                actual_prices.append({
                    'price': row[0],
                    'timestamp': row[1]
                })
            
            return actual_prices
            
        except Exception as e:
            print(f"Error fetching actual prices for {symbol}: {e}")
            return []
        finally:
            self.db.close_connection(conn)
    
    def calculate_price_accuracy(self, predicted_price, actual_prices):
        """Calculate price prediction accuracy metrics"""
        if not actual_prices:
            return {
                'has_data': False,
                'message': 'No actual price data available after decision timestamp'
            }
        
        actual_prices_list = [p['price'] for p in actual_prices]
        
        # Calculate various accuracy metrics
        mae = np.mean([abs(predicted_price - actual) for actual in actual_prices_list])
        mse = np.mean([(predicted_price - actual) ** 2 for actual in actual_prices_list])
        rmse = np.sqrt(mse)
        
        # Percentage error
        mape = np.mean([abs((predicted_price - actual) / actual) * 100 for actual in actual_prices_list])
        
        # Direction accuracy (if price went up/down as predicted)
        price_change = (actual_prices_list[-1] - actual_prices_list[0]) / actual_prices_list[0] * 100
        predicted_change = (predicted_price - actual_prices_list[0]) / actual_prices_list[0] * 100
        
        direction_correct = (price_change > 0 and predicted_change > 0) or (price_change < 0 and predicted_change < 0)
        
        return {
            'has_data': True,
            'hours_of_data': len(actual_prices_list),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'direction_correct': direction_correct,
            'actual_price_change': price_change,
            'predicted_price_change': predicted_change,
            'final_actual_price': actual_prices_list[-1],
            'initial_actual_price': actual_prices_list[0]
        }
    
    def calculate_signal_accuracy(self, signal, actual_prices):
        """Calculate signal accuracy for 5-level system: STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY"""
        if not actual_prices or len(actual_prices) < 2:
            return {
                'has_data': False,
                'message': 'Insufficient actual price data for signal validation'
            }
        
        initial_price = actual_prices[0]['price']
        final_price = actual_prices[-1]['price']
        price_change = (final_price - initial_price) / initial_price * 100
        
        # Determine if signal was correct based on 5-level system
        signal_correct = False
        signal_strength = "WEAK"
        
        if signal == "STRONG_BUY":
            signal_correct = price_change > 1.0  # At least 1% gain for strong buy
            signal_strength = "STRONG" if signal_correct else "WEAK"
        elif signal == "BUY":
            signal_correct = price_change > 0.5  # At least 0.5% gain for buy
            signal_strength = "MODERATE" if signal_correct else "WEAK"
        elif signal == "HOLD":
            signal_correct = abs(price_change) <= 0.5  # Within 0.5% range for hold
            signal_strength = "NEUTRAL"
        elif signal == "SELL":
            signal_correct = price_change < -0.5  # At least 0.5% loss for sell
            signal_strength = "MODERATE" if signal_correct else "WEAK"
        elif signal == "STRONG_SELL":
            signal_correct = price_change < -1.0  # At least 1% loss for strong sell
            signal_strength = "STRONG" if signal_correct else "WEAK"
        
        return {
            'has_data': True,
            'signal_correct': signal_correct,
            'actual_price_change': price_change,
            'signal': signal,
            'signal_strength': signal_strength,
            'hours_of_data': len(actual_prices)
        }
    
    def check_decision_validity(self, decision_timestamp):
        """Check if decision is still valid (within 12 hours)"""
        decision_time = datetime.fromisoformat(decision_timestamp.replace('Z', '+00:00'))
        current_time = datetime.utcnow()
        hours_passed = (current_time - decision_time).total_seconds() / 3600
        
        if hours_passed > 12:
            return {
                'is_valid': False,
                'hours_passed': hours_passed,
                'message': f'Decision is {hours_passed:.1f} hours old (>12 hours). New AI signal needed.'
            }
        else:
            return {
                'is_valid': True,
                'hours_passed': hours_passed,
                'message': f'Decision is {hours_passed:.1f} hours old (valid)'
            }
    
    def analyze_decision(self, decision):
        """Analyze a single AI decision"""
        symbol = decision['symbol']
        decision_timestamp = decision['decision_timestamp']
        signal = decision['signal']
        predicted_price = decision['predicted_price']
        
        print(f"\n📊 Analyzing decision for {symbol} at {decision_timestamp}")
        
        # Check decision validity
        validity = self.check_decision_validity(decision_timestamp)
        print(f"   Validity: {validity['message']}")
        
        if not validity['is_valid']:
            return {
                'symbol': symbol,
                'decision_timestamp': decision_timestamp,
                'validity': validity,
                'price_accuracy': {'has_data': False, 'message': 'Decision expired'},
                'signal_accuracy': {'has_data': False, 'message': 'Decision expired'}
            }
        
        # Get actual prices after decision
        actual_prices = self.get_actual_prices_after_decision(symbol, decision_timestamp)
        
        if not actual_prices:
            print(f"   ⚠️  No actual price data available after decision timestamp")
            return {
                'symbol': symbol,
                'decision_timestamp': decision_timestamp,
                'validity': validity,
                'price_accuracy': {'has_data': False, 'message': 'No actual data after decision'},
                'signal_accuracy': {'has_data': False, 'message': 'No actual data after decision'}
            }
        
        print(f"   📈 Found {len(actual_prices)} hours of actual price data")
        
        # Calculate accuracies
        price_accuracy = self.calculate_price_accuracy(predicted_price, actual_prices)
        signal_accuracy = self.calculate_signal_accuracy(signal, actual_prices)
        
        if price_accuracy['has_data']:
            print(f"   💰 Price Accuracy - MAE: {price_accuracy['mae']:.2f}, MAPE: {price_accuracy['mape']:.2f}%")
            print(f"   📊 Direction Correct: {price_accuracy['direction_correct']}")
        
        if signal_accuracy['has_data']:
            print(f"   🎯 Signal Accuracy: {signal_accuracy['signal_correct']} (actual change: {signal_accuracy['actual_price_change']:+.2f}%, strength: {signal_accuracy.get('signal_strength', 'N/A')})")
        
        return {
            'symbol': symbol,
            'decision_timestamp': decision_timestamp,
            'validity': validity,
            'price_accuracy': price_accuracy,
            'signal_accuracy': signal_accuracy
        }
    
    def calculate_overall_metrics(self, analysis_results):
        """Calculate overall accuracy metrics"""
        valid_decisions = [r for r in analysis_results if r['price_accuracy']['has_data']]
        
        if not valid_decisions:
            return {
                'total_decisions': len(analysis_results),
                'valid_decisions': 0,
                'message': 'No valid decisions with actual data available'
            }
        
        # Price accuracy metrics
        price_maes = [r['price_accuracy']['mae'] for r in valid_decisions]
        price_mapes = [r['price_accuracy']['mape'] for r in valid_decisions]
        direction_accuracies = [r['price_accuracy']['direction_correct'] for r in valid_decisions]
        
        # Signal accuracy metrics
        signal_accuracies = [r['signal_accuracy']['signal_correct'] for r in valid_decisions if r['signal_accuracy']['has_data']]
        
        return {
            'total_decisions': len(analysis_results),
            'valid_decisions': len(valid_decisions),
            'price_accuracy': {
                'avg_mae': np.mean(price_maes),
                'avg_mape': np.mean(price_mapes),
                'direction_accuracy': np.mean(direction_accuracies)
            },
            'signal_accuracy': {
                'accuracy_rate': np.mean(signal_accuracies) if signal_accuracies else 0,
                'total_signals': len(signal_accuracies)
            }
        }
    
    def run_accuracy_check(self, symbol=None, hours_back=24):
        """Run accuracy check for all or specific symbol"""
        print("🔍 Starting Refined AI Accuracy Check...")
        
        # Get AI decisions
        decisions = self.get_ai_decisions(symbol, hours_back)
        
        if not decisions:
            print("❌ No AI decisions found in the specified time range")
            return [], {
                'total_decisions': 0,
                'valid_decisions': 0,
                'message': 'No AI decisions found in the specified time range'
            }
        # Apply optional filters to avoid skewing averages
        if self.min_confidence is not None:
            decisions = [d for d in decisions if float(d.get('confidence_score', 0.0)) >= self.min_confidence]

        if self.signal_whitelist:
            wl = {s.upper() for s in self.signal_whitelist}
            decisions = [d for d in decisions if (d.get('signal','').upper() in wl)]

        if self.latest_per_symbol:
            latest_map = {}
            for d in decisions:
                key = d['symbol']
                ts = d['decision_timestamp']
                if key not in latest_map or ts > latest_map[key]['decision_timestamp']:
                    latest_map[key] = d
            decisions = list(latest_map.values())

        print(f"📊 Found {len(decisions)} AI decisions to analyze")
        
        # Analyze each decision
        analysis_results = []
        for decision in decisions:
            result = self.analyze_decision(decision)
            analysis_results.append(result)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(analysis_results)
        
        # Display results
        print("\n" + "="*60)
        print("📈 REFINED AI ACCURACY REPORT")
        print("="*60)
        
        print(f"Total Decisions Analyzed: {overall_metrics['total_decisions']}")
        print(f"Valid Decisions (with data): {overall_metrics['valid_decisions']}")
        
        if overall_metrics['valid_decisions'] > 0:
            print(f"\n💰 PRICE ACCURACY:")
            print(f"  Average MAE: {overall_metrics['price_accuracy']['avg_mae']:.2f}")
            print(f"  Average MAPE: {overall_metrics['price_accuracy']['avg_mape']:.2f}%")
            print(f"  Direction Accuracy: {overall_metrics['price_accuracy']['direction_accuracy']:.2%}")
            
            print(f"\n🎯 SIGNAL ACCURACY:")
            print(f"  Accuracy Rate: {overall_metrics['signal_accuracy']['accuracy_rate']:.2%}")
            print(f"  Total Signals: {overall_metrics['signal_accuracy']['total_signals']}")
        else:
            print(f"\n⚠️  {overall_metrics['message']}")
        
        # Save detailed results
        output_file = os.path.join(os.path.dirname(__file__), "accuracy_report.json")
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'overall_metrics': overall_metrics,
                'detailed_results': analysis_results
            }, f, indent=2)
        
        print(f"\n✅ Detailed results saved to {output_file}")
        
        return analysis_results, overall_metrics

def main():
    """Main execution function"""
    import argparse
    parser = argparse.ArgumentParser(description="Refined AI accuracy checker")
    parser.add_argument("--since-hours", type=int, default=24, help="Look back N hours (default 24)")
    parser.add_argument("--symbol", type=str, default=None, help="Limit to one symbol")
    parser.add_argument("--min-confidence", type=float, default=None, help="Only evaluate decisions with confidence >= X")
    parser.add_argument("--signals", type=str, default=None, help="Comma list of signals to include (e.g., BUY,SELL)")
    parser.add_argument("--latest-per-symbol", action="store_true", help="Only evaluate the latest decision per symbol")
    args = parser.parse_args()

    checker = RefinedAIAccuracyChecker()
    if args.min_confidence is not None:
        checker.min_confidence = args.min_confidence
    if args.signals:
        checker.signal_whitelist = {s.strip().upper() for s in args.signals.split(',') if s.strip()}
    if args.latest_per_symbol:
        checker.latest_per_symbol = True

    results, metrics = checker.run_accuracy_check(symbol=args.symbol, hours_back=args.since_hours)

if __name__ == "__main__":
    main()
