# AI Signals Evaluation System
# This module evaluates the accuracy and performance of AI trading signals

import os
import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

# Add parent directory to path for database access
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase

class AIEvaluation:
    def __init__(self):
        """Initialize the AI evaluation system"""
        self.db = TradeDatabase()
        self.evaluation_results = {}
        
    def get_ai_decisions(self, hours_back=24):
        """Get AI decisions from the last N hours"""
        try:
            conn, cursor = self.db.sql_connect()
            query = """
                SELECT symbol, action, action_code, confidence, reasoning, current_price, 
                       decision_timestamp, price_target, stop_loss, risk_reward_ratio
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
                    'symbol': row[0],
                    'action': row[1],
                    'action_code': row[2],
                    'confidence': row[3],
                    'reasoning': row[4],
                    'current_price': row[5],
                    'timestamp': row[6],
                    'price_target': row[7],
                    'stop_loss': row[8],
                    'risk_reward_ratio': row[9]
                })
            
            return decisions
            
        except Exception as e:
            print(f"Error fetching AI decisions: {e}")
            return []
    
    def get_price_data_after_decision(self, symbol, decision_timestamp, hours_forward=4):
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
            print(f"Error fetching price data for {symbol}: {e}")
            return []
    
    def calculate_directional_accuracy(self, decision, price_data):
        """Calculate if the AI decision was directionally correct"""
        if not price_data or len(price_data) < 2:
            return None, "Insufficient price data"
        
        entry_price = decision['current_price']
        final_price = price_data[-1]['close']
        
        price_change = (final_price - entry_price) / entry_price
        
        # Determine if the decision was correct
        action = decision['action']
        is_correct = False
        confidence_level = "Unknown"
        
        if action in ['STRONG_BUY', 'BUY']:
            is_correct = price_change > 0.001  # 0.1% threshold for buy signals
            confidence_level = "High" if action == 'STRONG_BUY' else "Medium"
        elif action in ['STRONG_SELL', 'SELL']:
            is_correct = price_change < -0.001  # 0.1% threshold for sell signals
            confidence_level = "High" if action == 'STRONG_SELL' else "Medium"
        elif action == 'HOLD':
            is_correct = abs(price_change) <= 0.001  # Within 0.1% for hold
            confidence_level = "Medium"
        
        return {
            'is_correct': is_correct,
            'price_change': price_change,
            'entry_price': entry_price,
            'final_price': final_price,
            'confidence_level': confidence_level,
            'action': action
        }
    
    def calculate_profit_loss(self, decision, price_data):
        """Calculate profit/loss if the decision was followed"""
        if not price_data or len(price_data) < 2:
            return None
        
        entry_price = decision['current_price']
        final_price = price_data[-1]['close']
        
        # Calculate P&L based on action
        action = decision['action']
        position_size = 1  # Assume 1 unit for calculation
        
        if action in ['STRONG_BUY', 'BUY']:
            pnl = (final_price - entry_price) * position_size
        elif action in ['STRONG_SELL', 'SELL']:
            pnl = (entry_price - final_price) * position_size
        else:  # HOLD
            pnl = 0
        
        return {
            'pnl': pnl,
            'pnl_percentage': (pnl / entry_price) * 100,
            'entry_price': entry_price,
            'exit_price': final_price,
            'action': action
        }
    
    def evaluate_decision(self, decision):
        """Evaluate a single AI decision"""
        symbol = decision['symbol']
        timestamp = decision['timestamp']
        
        # Get price data after the decision
        price_data = self.get_price_data_after_decision(symbol, timestamp, 4)  # 4 hours forward
        
        if not price_data:
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'action': decision['action'],
                'confidence': decision['confidence'],
                'status': 'No price data available',
                'directional_accuracy': None,
                'profit_loss': None
            }
        
        # Calculate directional accuracy
        directional_result = self.calculate_directional_accuracy(decision, price_data)
        
        # Calculate profit/loss
        pnl_result = self.calculate_profit_loss(decision, price_data)
        
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'action': decision['action'],
            'confidence': decision['confidence'],
            'status': 'Evaluated',
            'directional_accuracy': directional_result,
            'profit_loss': pnl_result,
            'price_data_points': len(price_data)
        }
    
    def evaluate_all_decisions(self, hours_back=24):
        """Evaluate all AI decisions from the last N hours"""
        print(f"🔍 Evaluating AI decisions from the last {hours_back} hours...")
        
        # Get all decisions
        decisions = self.get_ai_decisions(hours_back)
        
        if not decisions:
            print("❌ No AI decisions found for evaluation")
            return {}
        
        print(f"📊 Found {len(decisions)} decisions to evaluate")
        
        # Evaluate each decision
        evaluation_results = []
        for i, decision in enumerate(decisions):
            print(f"Evaluating decision {i+1}/{len(decisions)}: {decision['symbol']} - {decision['action']}")
            result = self.evaluate_decision(decision)
            evaluation_results.append(result)
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_statistics(evaluation_results)
        
        # Store results
        self.evaluation_results = {
            'individual_results': evaluation_results,
            'overall_statistics': overall_stats,
            'evaluation_timestamp': datetime.utcnow().isoformat(),
            'evaluation_period_hours': hours_back
        }
        
        # Save results to file
        self.save_evaluation_results()
        
        return self.evaluation_results
    
    def calculate_overall_statistics(self, evaluation_results):
        """Calculate overall performance statistics"""
        total_decisions = len(evaluation_results)
        evaluated_decisions = [r for r in evaluation_results if r['status'] == 'Evaluated']
        
        if not evaluated_decisions:
            return {
                'total_decisions': total_decisions,
                'evaluated_decisions': 0,
                'directional_accuracy': 0,
                'average_confidence': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'average_pnl_per_trade': 0
            }
        
        # Directional accuracy
        correct_predictions = [r for r in evaluated_decisions 
                             if r['directional_accuracy'] and r['directional_accuracy']['is_correct']]
        directional_accuracy = len(correct_predictions) / len(evaluated_decisions) * 100
        
        # Average confidence
        confidences = [r['confidence'] for r in evaluated_decisions if r['confidence']]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Profit/Loss analysis
        pnl_values = [r['profit_loss']['pnl'] for r in evaluated_decisions 
                     if r['profit_loss'] and r['profit_loss']['pnl'] is not None]
        total_pnl = sum(pnl_values) if pnl_values else 0
        
        profitable_trades = [pnl for pnl in pnl_values if pnl > 0] if pnl_values else []
        win_rate = len(profitable_trades) / len(pnl_values) * 100 if pnl_values else 0
        
        avg_pnl_per_trade = np.mean(pnl_values) if pnl_values else 0
        
        # Action distribution
        action_counts = {}
        for result in evaluated_decisions:
            action = result['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'evaluated_decisions': len(evaluated_decisions),
            'directional_accuracy': round(directional_accuracy, 2),
            'average_confidence': round(avg_confidence, 3),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'average_pnl_per_trade': round(avg_pnl_per_trade, 2),
            'action_distribution': action_counts,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(pnl_values) - len(profitable_trades) if pnl_values else 0
        }
    
    def save_evaluation_results(self):
        """Save evaluation results to JSON file"""
        output_path = os.path.join(os.path.dirname(__file__), "ai_evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        print(f"📁 Evaluation results saved to {output_path}")
    
    def print_evaluation_summary(self):
        """Print a summary of evaluation results"""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        stats = self.evaluation_results['overall_statistics']
        
        print("\n" + "="*60)
        print("📊 AI SIGNALS EVALUATION SUMMARY")
        print("="*60)
        print(f"📅 Evaluation Period: {self.evaluation_results['evaluation_period_hours']} hours")
        print(f"📈 Total Decisions: {stats['total_decisions']}")
        print(f"✅ Evaluated Decisions: {stats['evaluated_decisions']}")
        print(f"🎯 Directional Accuracy: {stats['directional_accuracy']}%")
        print(f"💪 Average Confidence: {stats['average_confidence']}")
        print(f"💰 Total P&L: ${stats['total_pnl']}")
        print(f"🏆 Win Rate: {stats['win_rate']}%")
        print(f"📊 Average P&L per Trade: ${stats['average_pnl_per_trade']}")
        print(f"✅ Profitable Trades: {stats['profitable_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        
        print("\n📋 Action Distribution:")
        for action, count in stats['action_distribution'].items():
            print(f"   {action}: {count}")
        
        # Performance assessment
        if stats['directional_accuracy'] >= 70:
            print(f"\n🎉 EXCELLENT! Accuracy of {stats['directional_accuracy']}% exceeds 70% target!")
        elif stats['directional_accuracy'] >= 60:
            print(f"\n👍 GOOD! Accuracy of {stats['directional_accuracy']}% is above 60%")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT! Accuracy of {stats['directional_accuracy']}% is below 60%")
        
        print("="*60)

if __name__ == "__main__":
    # Run evaluation
    evaluator = AIEvaluation()
    results = evaluator.evaluate_all_decisions(hours_back=24)
    evaluator.print_evaluation_summary()
