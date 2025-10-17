#!/usr/bin/env python3
"""
Percentage Accuracy Analyzer for Refined AI Signals
This script calculates percentage accuracy scores (out of 100%) for each symbol's predictions.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

class PercentageAccuracyAnalyzer:
    def __init__(self):
        self.db = TradeDatabase()
        
    def calculate_percentage_accuracy(self, symbol: str = None, days_back: int = 7) -> Tuple[List[Dict], Dict]:
        """
        Calculate percentage accuracy scores (out of 100%) for AI signals
        """
        print(f"🎯 Calculating Percentage Accuracy Scores (Last {days_back} days)")
        print("=" * 70)
        
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
                
            # Calculate percentage accuracy metrics
            accuracy_metrics = self.calculate_percentage_metrics(
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
                print(f"   ✅ Accuracy Score: {accuracy_metrics['accuracy_score']:.1f}%")
            else:
                print(f"   ❌ Could not calculate accuracy metrics")
        
        # Calculate overall statistics
        if accuracy_results:
            overall_stats = self.calculate_overall_percentage_stats(accuracy_results)
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
    
    def calculate_percentage_metrics(self, predicted_price: float, actual_prices: List[Dict], 
                                   recommendation: str, decision_time: datetime) -> Dict:
        """Calculate percentage accuracy metrics for a single decision"""
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
            
            # Calculate percentage accuracy score (out of 100%)
            # Higher accuracy = lower MAPE, so we invert it
            # Perfect accuracy (0% MAPE) = 100% score
            # 10% MAPE = 90% score, 20% MAPE = 80% score, etc.
            accuracy_score = max(0, 100 - mape)
            
            # Calculate directional accuracy
            price_change = first_actual - predicted_price
            price_change_pct = (price_change / predicted_price) * 100 if predicted_price != 0 else 0
            
            # Determine if directional prediction was correct
            if recommendation == "BUY" and price_change > 0:
                directional_correct = True
                directional_score = 100
            elif recommendation == "SELL" and price_change < 0:
                directional_correct = True
                directional_score = 100
            elif recommendation == "HOLD":
                # For HOLD, check if price stayed within 2% range
                if abs(price_change_pct) <= 2.0:
                    directional_correct = True
                    directional_score = 100
                else:
                    directional_correct = False
                    directional_score = max(0, 100 - abs(price_change_pct) * 10)
            else:
                directional_correct = False
                directional_score = 0
            
            # Calculate overall accuracy score (weighted average)
            # 70% weight on price accuracy, 30% weight on directional accuracy
            overall_accuracy = (accuracy_score * 0.7) + (directional_score * 0.3)
            
            # Calculate confidence-weighted accuracy
            # This considers the AI's confidence in its prediction
            confidence_weighted_accuracy = overall_accuracy * (decision_time.hour / 24)  # Simple time-based weight
            
            return {
                'actual_price': first_actual,
                'price_error': price_error,
                'mae': mae,
                'mape': mape,
                'accuracy_score': accuracy_score,
                'directional_correct': directional_correct,
                'directional_score': directional_score,
                'overall_accuracy': overall_accuracy,
                'confidence_weighted_accuracy': confidence_weighted_accuracy,
                'hours_available': hours_available,
                'price_change_pct': price_change_pct
            }
            
        except Exception as e:
            print(f"❌ Error calculating percentage metrics: {e}")
            return None
    
    def calculate_overall_percentage_stats(self, accuracy_results: List[Dict]) -> Dict:
        """Calculate overall percentage statistics from accuracy results"""
        if not accuracy_results:
            return {'total_decisions': 0, 'valid_decisions': 0}
        
        total_decisions = len(accuracy_results)
        accuracy_scores = [r['accuracy_score'] for r in accuracy_results]
        directional_scores = [r['directional_score'] for r in accuracy_results]
        overall_scores = [r['overall_accuracy'] for r in accuracy_results]
        confidence_weighted_scores = [r['confidence_weighted_accuracy'] for r in accuracy_results]
        
        # Calculate symbol-wise statistics
        symbol_stats = {}
        for result in accuracy_results:
            symbol = result['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = []
            symbol_stats[symbol].append(result)
        
        symbol_percentages = {}
        for symbol, results in symbol_stats.items():
            symbol_accuracy = [r['accuracy_score'] for r in results]
            symbol_directional = [r['directional_score'] for r in results]
            symbol_overall = [r['overall_accuracy'] for r in results]
            
            symbol_percentages[symbol] = {
                'decisions_count': len(results),
                'avg_accuracy_score': sum(symbol_accuracy) / len(symbol_accuracy),
                'avg_directional_score': sum(symbol_directional) / len(symbol_directional),
                'avg_overall_score': sum(symbol_overall) / len(symbol_overall),
                'best_accuracy': max(symbol_accuracy),
                'worst_accuracy': min(symbol_accuracy)
            }
        
        return {
            'total_decisions': total_decisions,
            'valid_decisions': total_decisions,
            'overall_accuracy_score': sum(accuracy_scores) / len(accuracy_scores),
            'overall_directional_score': sum(directional_scores) / len(directional_scores),
            'overall_combined_score': sum(overall_scores) / len(overall_scores),
            'overall_confidence_weighted_score': sum(confidence_weighted_scores) / len(confidence_weighted_scores),
            'best_accuracy': max(accuracy_scores),
            'worst_accuracy': min(accuracy_scores),
            'symbol_percentages': symbol_percentages
        }
    
    def print_percentage_report(self, accuracy_results: List[Dict], overall_stats: Dict):
        """Print a detailed percentage accuracy report"""
        print("\n" + "=" * 70)
        print("🎯 PERCENTAGE ACCURACY ANALYSIS REPORT (Out of 100%)")
        print("=" * 70)
        
        print(f"📊 Overall Performance:")
        print(f"   Total Decisions: {overall_stats['total_decisions']}")
        print(f"   Valid Decisions: {overall_stats['valid_decisions']}")
        
        if overall_stats['valid_decisions'] > 0:
            print(f"   🎯 Overall Accuracy Score: {overall_stats['overall_accuracy_score']:.1f}%")
            print(f"   📈 Overall Directional Score: {overall_stats['overall_directional_score']:.1f}%")
            print(f"   ⭐ Overall Combined Score: {overall_stats['overall_combined_score']:.1f}%")
            print(f"   🏆 Best Single Accuracy: {overall_stats['best_accuracy']:.1f}%")
            print(f"   📉 Worst Single Accuracy: {overall_stats['worst_accuracy']:.1f}%")
        
        print(f"\n📋 Symbol-by-Symbol Percentage Accuracy:")
        print("-" * 70)
        
        # Sort symbols by overall score (best first)
        symbol_scores = []
        for symbol, stats in overall_stats['symbol_percentages'].items():
            symbol_scores.append((symbol, stats['avg_overall_score']))
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (symbol, overall_score) in enumerate(symbol_scores, 1):
            stats = overall_stats['symbol_percentages'][symbol]
            print(f"\n{i}. {symbol}:")
            print(f"   🎯 Accuracy Score: {stats['avg_accuracy_score']:.1f}%")
            print(f"   📈 Directional Score: {stats['avg_directional_score']:.1f}%")
            print(f"   ⭐ Overall Score: {stats['avg_overall_score']:.1f}%")
            print(f"   📊 Decisions: {stats['decisions_count']}")
            print(f"   🏆 Best: {stats['best_accuracy']:.1f}% | 📉 Worst: {stats['worst_accuracy']:.1f}%")
            
            # Performance rating
            if stats['avg_overall_score'] >= 90:
                rating = "🏆 EXCELLENT"
            elif stats['avg_overall_score'] >= 80:
                rating = "✅ VERY GOOD"
            elif stats['avg_overall_score'] >= 70:
                rating = "👍 GOOD"
            elif stats['avg_overall_score'] >= 60:
                rating = "⚠️ FAIR"
            else:
                rating = "❌ NEEDS IMPROVEMENT"
            
            print(f"   📊 Rating: {rating}")
        
        print(f"\n📈 Recent High-Performance Decisions:")
        print("-" * 70)
        
        # Show top 5 recent decisions by accuracy score
        recent_decisions = sorted(accuracy_results, key=lambda x: x['accuracy_score'], reverse=True)[:5]
        
        for i, decision in enumerate(recent_decisions, 1):
            decision_time = datetime.fromisoformat(decision['decision_time'])
            print(f"{i}. {decision['symbol']} ({decision_time.strftime('%m-%d %H:%M')}): "
                  f"{decision['recommendation']} - {decision['accuracy_score']:.1f}% accuracy")

def main():
    analyzer = PercentageAccuracyAnalyzer()
    
    # Analyze all symbols for the last 7 days
    accuracy_results, overall_stats = analyzer.calculate_percentage_accuracy(days_back=7)
    
    # Print detailed report
    analyzer.print_percentage_report(accuracy_results, overall_stats)
    
    # Save results
    results_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_stats': overall_stats,
        'detailed_results': accuracy_results
    }
    
    output_file = 'percentage_accuracy_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
