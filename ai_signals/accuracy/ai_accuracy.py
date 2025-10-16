#!/usr/bin/env python3
"""
AI Signals Accuracy Calculator - Simplified All-in-One

Combines all accuracy testing into a single file with different modes:
- python ai_accuracy.py real    -> Real-time accuracy
- python ai_accuracy.py monitor -> Monitoring mode
- python ai_accuracy.py daily   -> Daily accuracy
- python ai_accuracy.py weekly  -> Weekly accuracy
- python ai_accuracy.py status  -> Quick status check
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db.database import TradeDatabase

class AIAccuracyCalculator:
    def __init__(self):
        """Initialize the accuracy calculator"""
        self.conn = None
        self.cursor = None
        
    def connect_db(self):
        """Connect to database"""
        try:
            self.conn, self.cursor = TradeDatabase.sql_connect()
            if self.conn is None:
                raise RuntimeError("Could not connect to SQLite database")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            TradeDatabase.close_connection(self.conn)
    
    def get_ai_decisions(self, hours_back: int = 24) -> List[Tuple]:
        """Get AI decisions from database"""
        if not self.cursor:
            return []
        
        try:
            query = """
                SELECT symbol, action, current_price, decision_timestamp, confidence, reasoning
                FROM ai_decisions 
                WHERE datetime(decision_timestamp) >= datetime('now', '-{} hours')
                ORDER BY decision_timestamp DESC
            """.format(hours_back)
            
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"❌ Error fetching AI decisions: {e}")
            return []
    
    def get_price_after_decision(self, symbol: str, decision_time: str, hours_later: int = 4) -> Optional[float]:
        """Get price after decision for accuracy calculation"""
        if not self.cursor:
            return None
        
        try:
            query = """
                SELECT close FROM historical_data_1h 
                WHERE symbol = ? 
                AND datetime(timestamp) > datetime(?)
                ORDER BY timestamp ASC 
                LIMIT ?
            """
            
            self.cursor.execute(query, (symbol, decision_time, hours_later))
            result = self.cursor.fetchone()
            return float(result[0]) if result else None
        except Exception as e:
            print(f"❌ Error fetching price data: {e}")
            return None
    
    def calculate_accuracy(self, decisions: List[Tuple], hours_later: int = 4) -> Dict:
        """Calculate accuracy metrics"""
        if not decisions:
            return {
                'total_decisions': 0,
                'evaluated_decisions': 0,
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'symbol_breakdown': {}
            }
        
        total_decisions = len(decisions)
        evaluated_decisions = 0
        correct_predictions = 0
        total_pnl = 0.0
        pnl_values = []
        symbol_stats = {}
        
        for decision in decisions:
            symbol, action, current_price, decision_time, confidence, reasoning = decision
            
            # Initialize symbol stats
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'total': 0,
                    'correct': 0,
                    'pnl': 0.0,
                    'confidence_sum': 0.0
                }
            
            symbol_stats[symbol]['total'] += 1
            symbol_stats[symbol]['confidence_sum'] += float(confidence or 0)
            
            try:
                current_price_f = float(current_price)
                
                # Get price after decision
                future_price = self.get_price_after_decision(symbol, decision_time, hours_later)
                
                if future_price is None:
                    continue
                
                evaluated_decisions += 1
                
                # Calculate price change
                price_change = (future_price - current_price_f) / current_price_f
                pnl = price_change * 10000  # Assuming $10k position size
                
                total_pnl += pnl
                pnl_values.append(pnl)
                symbol_stats[symbol]['pnl'] += pnl
                
                # Determine if prediction was correct
                is_correct = False
                if action in ['STRONG_BUY', 'BUY']:
                    is_correct = price_change > 0.001  # 0.1% threshold
                elif action in ['STRONG_SELL', 'SELL']:
                    is_correct = price_change < -0.001
                elif action == 'HOLD':
                    is_correct = abs(price_change) <= 0.001
                
                if is_correct:
                    correct_predictions += 1
                    symbol_stats[symbol]['correct'] += 1
                
            except (ValueError, TypeError) as e:
                continue
        
        # Calculate overall metrics
        accuracy = (correct_predictions / evaluated_decisions * 100) if evaluated_decisions > 0 else 0.0
        win_rate = len([p for p in pnl_values if p > 0]) / len(pnl_values) * 100 if pnl_values else 0.0
        
        # Calculate symbol-level accuracy
        symbol_breakdown = {}
        for symbol, stats in symbol_stats.items():
            if stats['total'] > 0:
                symbol_breakdown[symbol] = {
                    'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0,
                    'total_decisions': stats['total'],
                    'correct_decisions': stats['correct'],
                    'avg_confidence': stats['confidence_sum'] / stats['total'],
                    'pnl': stats['pnl']
                }
        
        return {
            'total_decisions': total_decisions,
            'evaluated_decisions': evaluated_decisions,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'symbol_breakdown': symbol_breakdown,
            'profitable_trades': len([p for p in pnl_values if p > 0]),
            'losing_trades': len([p for p in pnl_values if p <= 0])
        }
    
    def print_accuracy_report(self, results: Dict, mode: str, hours_back: int):
        """Print formatted accuracy report"""
        print(f"\n📊 AI SIGNALS ACCURACY REPORT - {mode.upper()}")
        print("=" * 70)
        print(f"⏰ Time Period: Last {hours_back} hours")
        print(f"🕐 Calculated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 70)
        
        if results['total_decisions'] == 0:
            print("❌ No AI decisions found in the specified time period")
            print("💡 Run 'python ai_realtime.py' to generate signals first")
            return
        
        print(f"📈 Total Decisions: {results['total_decisions']}")
        print(f"✅ Evaluated Decisions: {results['evaluated_decisions']}")
        
        if results['evaluated_decisions'] == 0:
            print("⏳ No decisions can be evaluated yet (need 4+ hours of future data)")
            print("💡 Wait 4+ hours after generating signals, then run again")
            return
        
        print(f"🎯 Directional Accuracy: {results['accuracy']:.1f}%")
        print(f"💰 Total P&L: ${results['total_pnl']:.2f}")
        print(f"🏆 Win Rate: {results['win_rate']:.1f}%")
        print(f"✅ Profitable Trades: {results['profitable_trades']}")
        print(f"❌ Losing Trades: {results['losing_trades']}")
        
        # Performance assessment
        if results['accuracy'] >= 70:
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print(f"🎉 EXCELLENT! Accuracy of {results['accuracy']:.1f}% exceeds 70% target!")
        elif results['accuracy'] >= 60:
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print(f"👍 GOOD! Accuracy of {results['accuracy']:.1f}% is above 60%")
        else:
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print(f"⚠️  NEEDS IMPROVEMENT! Accuracy of {results['accuracy']:.1f}% is below 60%")
        
        # Symbol breakdown
        if results['symbol_breakdown']:
            print(f"\n📊 SYMBOL PERFORMANCE:")
            for symbol, stats in results['symbol_breakdown'].items():
                print(f"   {symbol}: {stats['accuracy']:.1f}% accuracy ({stats['correct_decisions']}/{stats['total_decisions']}) conf: {stats['avg_confidence']:.2f}")
    
    def run_real_time_accuracy(self):
        """Run real-time accuracy calculation (last 6 hours)"""
        if not self.connect_db():
            return False
        
        try:
            decisions = self.get_ai_decisions(6)
            results = self.calculate_accuracy(decisions, 4)
            self.print_accuracy_report(results, "real-time", 6)
            return True
        finally:
            self.close_db()
    
    def run_daily_accuracy(self):
        """Run daily accuracy calculation (last 24 hours)"""
        if not self.connect_db():
            return False
        
        try:
            decisions = self.get_ai_decisions(24)
            results = self.calculate_accuracy(decisions, 4)
            self.print_accuracy_report(results, "daily", 24)
            return True
        finally:
            self.close_db()
    
    def run_weekly_accuracy(self):
        """Run weekly accuracy calculation (last 168 hours)"""
        if not self.connect_db():
            return False
        
        try:
            decisions = self.get_ai_decisions(168)
            results = self.calculate_accuracy(decisions, 4)
            self.print_accuracy_report(results, "weekly", 168)
            return True
        finally:
            self.close_db()
    
    def run_monitoring_mode(self):
        """Run monitoring mode (continuous updates)"""
        print("🔍 AI SIGNALS MONITORING MODE")
        print("=" * 40)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 40)
        
        try:
            while True:
                print(f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                self.run_real_time_accuracy()
                
                print("\n⏳ Waiting 5 minutes for next update...")
                import time
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            return True
    
    def run_status_check(self):
        """Run quick status check"""
        if not self.connect_db():
            return False
        
        try:
            # Check AI decisions
            decisions = self.get_ai_decisions(24)
            print(f"📊 AI DECISIONS STATUS")
            print("=" * 30)
            print(f"Total decisions (24h): {len(decisions)}")
            
            if decisions:
                latest = decisions[0]
                print(f"Latest decision: {latest[0]} - {latest[1]} at {latest[3]}")
                print(f"Latest confidence: {latest[4]:.2f}")
            
            # Check data availability
            self.cursor.execute("SELECT COUNT(*) FROM historical_data_1h")
            hist_count = self.cursor.fetchone()[0]
            print(f"Historical data points: {hist_count}")
            
            if hist_count > 0:
                self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM historical_data_1h")
                min_ts, max_ts = self.cursor.fetchone()
                print(f"Data range: {min_ts} to {max_ts}")
            
            print("\n💡 Status: Ready for accuracy calculation" if len(decisions) > 0 else "⚠️  No decisions found - run 'python ai_realtime.py' first")
            return True
            
        finally:
            self.close_db()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AI Signals Accuracy Calculator')
    parser.add_argument('mode', choices=['real', 'daily', 'weekly', 'monitor', 'status'], 
                       help='Accuracy calculation mode')
    
    args = parser.parse_args()
    
    calculator = AIAccuracyCalculator()
    
    if args.mode == 'real':
        calculator.run_real_time_accuracy()
    elif args.mode == 'daily':
        calculator.run_daily_accuracy()
    elif args.mode == 'weekly':
        calculator.run_weekly_accuracy()
    elif args.mode == 'monitor':
        calculator.run_monitoring_mode()
    elif args.mode == 'status':
        calculator.run_status_check()

if __name__ == "__main__":
    main()

