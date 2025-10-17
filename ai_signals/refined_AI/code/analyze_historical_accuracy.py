#!/usr/bin/env python3
"""
Historical Accuracy Analyzer for Refined AI Signals
This script analyzes AI signal accuracy regardless of age for historical analysis.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

class HistoricalAccuracyAnalyzer:
    def __init__(self):
        self.db = TradeDatabase()
        
    def analyze_historical_accuracy(self, symbol: str = None, days_back: int = 7) -> Tuple[List[Dict], Dict]:
        """
        Analyze historical accuracy of AI signals regardless of age
        """
        print(f"🔍 Analyzing Historical Accuracy (Last {days_back} days)")
        print("=" * 60)
        
        # Get AI decisions
        decisions = self.get_ai_decisions(symbol, days_back)
        if not decisions:
            print("❌ No AI decisions found in the specified time range")
            return [], {'total_decisions': 0, 'valid_decisions': 0, 'message': 'No AI decisions found'}
        
        print(f"📊 Found {len(decisions)} AI decisions to analyze")
        
        accuracy_results = []
        valid_count = 0
        
        for decision in decisions:
            symbol = decision['symbol']
            decision_time = datetime.fromisoformat(decision['timestamp'].replace('Z', '+00:00'))
            predicted_price = decision['predicted_price']
            recommendation = decision['recommendation']
            
            print(f"\n📊 Analyzing {symbol} at {decision_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get actual prices after the decision time
            actual_prices = self.get_actual_prices_after(symbol, decision_time)
            
            if not actual_prices:
                print(f"   ⚠️  No actual data available after decision time")
                continue
                
            # Calculate accuracy metrics
            accuracy_metrics = self.calculate_accuracy_metrics(
                predicted_price, actual_prices, recommendation, decision_time
            )
            
            if accuracy_metrics:
                accuracy_results.append({
                    'symbol': symbol,
                    'decision_time': decision_time.isoformat(),
                    'predicted_price': predicted_price,
                    'recommendation': recommendation,
                    'confidence': decision.get('confidence', 0.0),
                    **accuracy_metrics
                })
                valid_count += 1
                print(f"   ✅ Accuracy: {accuracy_metrics['mae']:.2f} MAE, {accuracy_metrics['mape']:.1f}% MAPE")
            else:
                print(f"   ❌ Could not calculate accuracy metrics")
        
        # Calculate overall statistics
        if accuracy_results:
            overall_stats = self.calculate_overall_stats(accuracy_results)
        else:
            overall_stats = {
                'total_decisions': len(decisions),
                'valid_decisions': 0,
                'message': 'No valid accuracy calculations possible'
            }
        
        return accuracy_results, overall_stats
    
    def get_ai_decisions(self, symbol: str = None, days_back: int = 7) -> List[Dict]:
        """Get AI decisions from the database"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            if symbol:
                query = """
                    SELECT symbol, decision_timestamp, predicted_price, signal, confidence_score, reasoning
                    FROM Smart_AI_decisions 
                    WHERE symbol = ? AND decision_timestamp >= ? AND decision_timestamp <= ?
                    ORDER BY decision_timestamp DESC
                """
                cursor.execute(query, (symbol, start_time.isoformat(), end_time.isoformat()))
            else:
                query = """
                    SELECT symbol, decision_timestamp, predicted_price, signal, confidence_score, reasoning
                    FROM Smart_AI_decisions 
                    WHERE decision_timestamp >= ? AND decision_timestamp <= ?
                    ORDER BY decision_timestamp DESC
                """
                cursor.execute(query, (start_time.isoformat(), end_time.isoformat()))
            
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            decisions = []
            for row in rows:
                decisions.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'predicted_price': row[2],
                    'recommendation': row[3],
                    'confidence': row[4],
                    'reason': row[5]
                })
            
            return decisions
            
        except Exception as e:
            print(f"❌ Error fetching AI decisions: {e}")
            return []
    
    def get_actual_prices_after(self, symbol: str, decision_time: datetime) -> List[Dict]:
        """Get actual prices after the decision time"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Get actual prices from historical_data_1h table
            query = """
                SELECT timestamp, close, high, low, open, volume
                FROM historical_data_1h 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT 24
            """
            cursor.execute(query, (symbol, decision_time.isoformat()))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            prices = []
            for row in rows:
                prices.append({
                    'timestamp': row[0],
                    'close': row[1],
                    'high': row[2],
                    'low': row[3],
                    'open': row[4],
                    'volume': row[5]
                })
            
            return prices
            
        except Exception as e:
            print(f"❌ Error fetching actual prices for {symbol}: {e}")
            return []
    
    def calculate_accuracy_metrics(self, predicted_price: float, actual_prices: List[Dict], 
                                 recommendation: str, decision_time: datetime) -> Dict:
        """Calculate accuracy metrics for a single decision"""
        if not actual_prices:
            return None
        
        try:
            # Get the first few actual prices for comparison
            first_actual = actual_prices[0]['close']
            hours_available = len(actual_prices)
            
            # Calculate basic accuracy metrics
            price_error = abs(predicted_price - first_actual)
            mae = price_error
            mape = (price_error / first_actual) * 100 if first_actual != 0 else 0
            
            # Calculate directional accuracy
            price_change = first_actual - predicted_price
            if recommendation == "BUY" and price_change > 0:
                directional_correct = True
            elif recommendation == "SELL" and price_change < 0:
                directional_correct = True
            elif recommendation == "HOLD":
                directional_correct = abs(price_change) < (predicted_price * 0.02)  # Within 2%
            else:
                directional_correct = False
            
            return {
                'actual_price': first_actual,
                'price_error': price_error,
                'mae': mae,
                'mape': mape,
                'directional_correct': directional_correct,
                'hours_available': hours_available,
                'price_change_pct': (price_change / predicted_price) * 100
            }
            
        except Exception as e:
            print(f"❌ Error calculating accuracy metrics: {e}")
            return None
    
    def calculate_overall_stats(self, accuracy_results: List[Dict]) -> Dict:
        """Calculate overall statistics from accuracy results"""
        if not accuracy_results:
            return {'total_decisions': 0, 'valid_decisions': 0}
        
        total_decisions = len(accuracy_results)
        mae_values = [r['mae'] for r in accuracy_results]
        mape_values = [r['mape'] for r in accuracy_results]
        directional_correct = sum(1 for r in accuracy_results if r['directional_correct'])
        
        return {
            'total_decisions': total_decisions,
            'valid_decisions': total_decisions,
            'avg_mae': sum(mae_values) / len(mae_values),
            'avg_mape': sum(mape_values) / len(mape_values),
            'directional_accuracy': (directional_correct / total_decisions) * 100,
            'best_mae': min(mae_values),
            'worst_mae': max(mae_values),
            'best_mape': min(mape_values),
            'worst_mape': max(mape_values)
        }
    
    def print_detailed_report(self, accuracy_results: List[Dict], overall_stats: Dict):
        """Print a detailed accuracy report"""
        print("\n" + "=" * 60)
        print("📈 HISTORICAL ACCURACY ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Decisions: {overall_stats['total_decisions']}")
        print(f"   Valid Decisions: {overall_stats['valid_decisions']}")
        
        if overall_stats['valid_decisions'] > 0:
            print(f"   Average MAE: {overall_stats['avg_mae']:.2f}")
            print(f"   Average MAPE: {overall_stats['avg_mape']:.1f}%")
            print(f"   Directional Accuracy: {overall_stats['directional_accuracy']:.1f}%")
            print(f"   Best MAE: {overall_stats['best_mae']:.2f}")
            print(f"   Worst MAE: {overall_stats['worst_mae']:.2f}")
        
        print(f"\n📋 Detailed Results by Symbol:")
        print("-" * 60)
        
        # Group by symbol
        symbol_results = {}
        for result in accuracy_results:
            symbol = result['symbol']
            if symbol not in symbol_results:
                symbol_results[symbol] = []
            symbol_results[symbol].append(result)
        
        for symbol, results in symbol_results.items():
            print(f"\n{symbol}:")
            symbol_mae = [r['mae'] for r in results]
            symbol_mape = [r['mape'] for r in results]
            symbol_directional = sum(1 for r in results if r['directional_correct'])
            
            print(f"   Decisions: {len(results)}")
            print(f"   Avg MAE: {sum(symbol_mae)/len(symbol_mae):.2f}")
            print(f"   Avg MAPE: {sum(symbol_mape)/len(symbol_mape):.1f}%")
            print(f"   Directional Accuracy: {(symbol_directional/len(results))*100:.1f}%")
            
            # Show recent decisions
            print(f"   Recent Decisions:")
            for result in results[:3]:  # Show last 3
                decision_time = datetime.fromisoformat(result['decision_time'])
                print(f"     {decision_time.strftime('%m-%d %H:%M')}: {result['recommendation']} "
                      f"({result['predicted_price']:.2f} → {result['actual_price']:.2f}) "
                      f"MAE: {result['mae']:.2f}")

def main():
    analyzer = HistoricalAccuracyAnalyzer()
    
    # Analyze all symbols for the last 7 days
    accuracy_results, overall_stats = analyzer.analyze_historical_accuracy(days_back=7)
    
    # Print detailed report
    analyzer.print_detailed_report(accuracy_results, overall_stats)
    
    # Save results
    results_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_stats': overall_stats,
        'detailed_results': accuracy_results
    }
    
    output_file = 'historical_accuracy_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
