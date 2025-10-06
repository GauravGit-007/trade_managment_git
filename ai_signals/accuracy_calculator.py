# AI Signals Accuracy Calculator
# This file directly queries the database and shows real-time accuracy rates

import sqlite3
import os
import sys
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append('..')
from db.database import TradeDatabase

class AccuracyCalculator:
    def __init__(self):
        """Initialize the accuracy calculator"""
        self.db = TradeDatabase()
        
    def get_ai_decisions(self, hours_back=24):
        """Get AI decisions from the database"""
        try:
            conn, cursor = self.db.sql_connect()
            query = """
                SELECT id, symbol, action, action_code, confidence, current_price, 
                       decision_timestamp, reasoning
                FROM ai_decisions 
                WHERE datetime(decision_timestamp) >= datetime('now', '-{} hours')
                ORDER BY decision_timestamp DESC
            """.format(hours_back)
            
            cursor.execute(query)
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            decisions = []
            for row in rows:
                decisions.append({
                    'id': row[0],
                    'symbol': row[1],
                    'action': row[2],
                    'action_code': row[3],
                    'confidence': row[4],
                    'current_price': row[5],
                    'timestamp': row[6],
                    'reasoning': row[7]
                })
            
            return decisions
            
        except Exception as e:
            print(f"❌ Error fetching AI decisions: {e}")
            return []
    
    def get_price_after_decision(self, symbol, decision_timestamp, hours_forward=4):
        """Get price data after a decision was made"""
        try:
            conn, cursor = self.db.sql_connect()
            query = """
                SELECT close, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                AND datetime(timestamp) > datetime(?)
                ORDER BY timestamp ASC
                LIMIT ?
            """
            cursor.execute(query, (symbol, decision_timestamp, hours_forward))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            return [{'close': row[0], 'timestamp': row[1]} for row in rows]
            
        except Exception as e:
            print(f"❌ Error fetching price data for {symbol}: {e}")
            return []
    
    def calculate_decision_accuracy(self, decision):
        """Calculate accuracy for a single decision"""
        symbol = decision['symbol']
        timestamp = decision['timestamp']
        action = decision['action']
        entry_price = decision['current_price']
        
        # Get price data after the decision
        price_data = self.get_price_after_decision(symbol, timestamp, 4)
        
        if not price_data or len(price_data) < 2:
            return {
                'symbol': symbol,
                'action': action,
                'status': 'Insufficient data',
                'is_correct': None,
                'price_change': None,
                'entry_price': entry_price,
                'final_price': None,
                'hours_forward': len(price_data)
            }
        
        # Calculate price change
        final_price = price_data[-1]['close']
        price_change = (final_price - entry_price) / entry_price
        
        # Determine if decision was correct
        is_correct = False
        if action in ['STRONG_BUY', 'BUY']:
            is_correct = price_change > 0.001  # 0.1% threshold
        elif action in ['STRONG_SELL', 'SELL']:
            is_correct = price_change < -0.001
        elif action == 'HOLD':
            is_correct = abs(price_change) <= 0.001
        
        return {
            'symbol': symbol,
            'action': action,
            'status': 'Evaluated',
            'is_correct': is_correct,
            'price_change': price_change,
            'entry_price': entry_price,
            'final_price': final_price,
            'hours_forward': len(price_data),
            'confidence': decision['confidence']
        }
    
    def calculate_overall_accuracy(self, hours_back=24):
        """Calculate overall accuracy for all decisions"""
        print(f"🔍 Calculating accuracy for last {hours_back} hours...")
        
        # Get all decisions
        decisions = self.get_ai_decisions(hours_back)
        
        if not decisions:
            print("❌ No AI decisions found in the specified time period")
            return None
        
        print(f"📊 Found {len(decisions)} decisions to evaluate")
        
        # Evaluate each decision
        results = []
        for i, decision in enumerate(decisions):
            print(f"   Evaluating {i+1}/{len(decisions)}: {decision['symbol']} - {decision['action']}")
            result = self.calculate_decision_accuracy(decision)
            results.append(result)
        
        # Calculate statistics
        evaluated_results = [r for r in results if r['status'] == 'Evaluated']
        
        if not evaluated_results:
            print("❌ No decisions could be evaluated (insufficient price data)")
            return None
        
        # Calculate metrics
        total_decisions = len(decisions)
        evaluated_count = len(evaluated_results)
        correct_predictions = sum(1 for r in evaluated_results if r['is_correct'])
        
        accuracy = (correct_predictions / evaluated_count) * 100 if evaluated_count > 0 else 0
        
        # Calculate confidence metrics
        confidences = [r['confidence'] for r in evaluated_results if r['confidence']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate P&L
        pnl_values = []
        for result in evaluated_results:
            if result['price_change'] is not None:
                if result['action'] in ['STRONG_BUY', 'BUY']:
                    pnl = result['price_change'] * 10000  # Assume $10k position
                elif result['action'] in ['STRONG_SELL', 'SELL']:
                    pnl = -result['price_change'] * 10000
                else:  # HOLD
                    pnl = 0
                pnl_values.append(pnl)
        
        total_pnl = sum(pnl_values) if pnl_values else 0
        profitable_trades = sum(1 for pnl in pnl_values if pnl > 0)
        win_rate = (profitable_trades / len(pnl_values)) * 100 if pnl_values else 0
        
        # Symbol breakdown
        symbol_stats = {}
        for result in evaluated_results:
            symbol = result['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'total': 0, 'correct': 0, 'confidence_sum': 0}
            symbol_stats[symbol]['total'] += 1
            if result['is_correct']:
                symbol_stats[symbol]['correct'] += 1
            symbol_stats[symbol]['confidence_sum'] += result['confidence']
        
        # Calculate symbol accuracies
        symbol_accuracy = {}
        for symbol, stats in symbol_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_conf = stats['confidence_sum'] / stats['total']
            symbol_accuracy[symbol] = {
                'accuracy': round(accuracy, 2),
                'total': stats['total'],
                'correct': stats['correct'],
                'avg_confidence': round(avg_conf, 3)
            }
        
        return {
            'total_decisions': total_decisions,
            'evaluated_decisions': evaluated_count,
            'accuracy': round(accuracy, 2),
            'correct_predictions': correct_predictions,
            'average_confidence': round(avg_confidence, 3),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'profitable_trades': profitable_trades,
            'losing_trades': len(pnl_values) - profitable_trades,
            'symbol_accuracy': symbol_accuracy,
            'individual_results': results,
            'calculation_time': datetime.utcnow().isoformat()
        }
    
    def print_accuracy_report(self, hours_back=24):
        """Print a comprehensive accuracy report"""
        print("=" * 70)
        print("📊 AI SIGNALS ACCURACY REPORT")
        print("=" * 70)
        print(f"⏰ Time Period: Last {hours_back} hours")
        print(f"🕐 Calculated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 70)
        
        # Calculate accuracy
        results = self.calculate_overall_accuracy(hours_back)
        
        if not results:
            print("❌ No data available for accuracy calculation")
            return
        
        # Print overall statistics
        print(f"📈 Total Decisions: {results['total_decisions']}")
        print(f"✅ Evaluated Decisions: {results['evaluated_decisions']}")
        print(f"🎯 Directional Accuracy: {results['accuracy']}%")
        print(f"💪 Average Confidence: {results['average_confidence']}")
        print(f"💰 Total P&L: ${results['total_pnl']}")
        print(f"🏆 Win Rate: {results['win_rate']}%")
        print(f"✅ Profitable Trades: {results['profitable_trades']}")
        print(f"❌ Losing Trades: {results['losing_trades']}")
        
        # Print symbol breakdown
        if results['symbol_accuracy']:
            print(f"\n📊 SYMBOL PERFORMANCE:")
            for symbol, stats in results['symbol_accuracy'].items():
                print(f"   {symbol}: {stats['accuracy']}% accuracy "
                      f"({stats['correct']}/{stats['total']}) "
                      f"conf: {stats['avg_confidence']}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if results['accuracy'] >= 70:
            print(f"🎉 EXCELLENT! Accuracy of {results['accuracy']}% exceeds 70% target!")
        elif results['accuracy'] >= 60:
            print(f"👍 GOOD! Accuracy of {results['accuracy']}% is above 60%")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT! Accuracy of {results['accuracy']}% is below 60%")
        
        print("=" * 70)
        
        # Save results to file
        self.save_accuracy_results(results)
        
        return results
    
    def save_accuracy_results(self, results):
        """Save accuracy results to JSON file"""
        output_path = os.path.join(os.path.dirname(__file__), "accuracy_report.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📁 Accuracy report saved to {output_path}")
    
    def get_live_accuracy(self):
        """Get live accuracy for recent decisions"""
        return self.print_accuracy_report(hours_back=6)
    
    def get_daily_accuracy(self):
        """Get daily accuracy report"""
        return self.print_accuracy_report(hours_back=24)
    
    def get_weekly_accuracy(self):
        """Get weekly accuracy report"""
        return self.print_accuracy_report(hours_back=168)  # 7 days

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Signals Accuracy Calculator')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Hours to look back for accuracy calculation')
    parser.add_argument('--live', action='store_true', 
                       help='Show live accuracy (last 6 hours)')
    parser.add_argument('--daily', action='store_true', 
                       help='Show daily accuracy (last 24 hours)')
    parser.add_argument('--weekly', action='store_true', 
                       help='Show weekly accuracy (last 7 days)')
    
    args = parser.parse_args()
    
    calculator = AccuracyCalculator()
    
    if args.live:
        calculator.get_live_accuracy()
    elif args.daily:
        calculator.get_daily_accuracy()
    elif args.weekly:
        calculator.get_weekly_accuracy()
    else:
        calculator.print_accuracy_report(hours_back=args.hours)

if __name__ == "__main__":
    main()
