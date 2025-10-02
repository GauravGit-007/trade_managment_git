# AI Signals Monitoring System
# This module provides real-time monitoring and alerting for AI trading signals

import os
import json
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# Add parent directory to path for database access
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase

class AIMonitoring:
    def __init__(self):
        """Initialize the AI monitoring system"""
        self.db = TradeDatabase()
        self.performance_history = deque(maxlen=100)  # Keep last 100 evaluations
        self.alert_thresholds = {
            'min_accuracy': 60.0,  # Minimum acceptable accuracy
            'max_confidence_drop': 0.1,  # Maximum confidence drop
            'max_consecutive_losses': 5,  # Maximum consecutive losing trades
            'min_win_rate': 50.0  # Minimum win rate
        }
        
    def get_recent_performance(self, hours_back=6):
        """Get recent AI performance metrics"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Get recent decisions with their evaluation
            query = """
                SELECT ad.symbol, ad.action, ad.confidence, ad.decision_timestamp,
                       ad.current_price, ad.price_target, ad.stop_loss
                FROM ai_decisions ad
                WHERE datetime(ad.decision_timestamp) >= datetime('now', '-{} hours')
                ORDER BY ad.decision_timestamp DESC
            """.format(hours_back)
            
            cursor.execute(query)
            decisions = cursor.fetchall()
            self.db.close_connection(conn)
            
            return [{
                'symbol': d[0],
                'action': d[1], 
                'confidence': d[2],
                'timestamp': d[3],
                'current_price': d[4],
                'price_target': d[5],
                'stop_loss': d[6]
            } for d in decisions]
            
        except Exception as e:
            print(f"Error fetching recent performance: {e}")
            return []
    
    def calculate_real_time_accuracy(self, decisions):
        """Calculate real-time accuracy for recent decisions"""
        if not decisions:
            return None
        
        accuracy_results = []
        
        for decision in decisions:
            # Get price data after decision
            price_data = self.get_price_data_after_decision(
                decision['symbol'], 
                decision['timestamp'], 
                hours_forward=4
            )
            
            if not price_data:
                continue
            
            # Calculate if decision was correct
            entry_price = decision['current_price']
            final_price = price_data[-1]['close']
            price_change = (final_price - entry_price) / entry_price
            
            action = decision['action']
            is_correct = False
            
            if action in ['STRONG_BUY', 'BUY']:
                is_correct = price_change > 0.001
            elif action in ['STRONG_SELL', 'SELL']:
                is_correct = price_change < -0.001
            elif action == 'HOLD':
                is_correct = abs(price_change) <= 0.001
            
            accuracy_results.append({
                'symbol': decision['symbol'],
                'action': action,
                'is_correct': is_correct,
                'price_change': price_change,
                'confidence': decision['confidence']
            })
        
        if not accuracy_results:
            return None
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for r in accuracy_results if r['is_correct'])
        total_predictions = len(accuracy_results)
        accuracy = (correct_predictions / total_predictions) * 100
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in accuracy_results])
        
        return {
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'average_confidence': avg_confidence,
            'individual_results': accuracy_results
        }
    
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
    
    def check_performance_alerts(self, performance_data):
        """Check for performance alerts and warnings"""
        alerts = []
        
        if not performance_data:
            alerts.append({
                'type': 'WARNING',
                'message': 'No recent performance data available',
                'timestamp': datetime.utcnow().isoformat()
            })
            return alerts
        
        accuracy = performance_data['accuracy']
        avg_confidence = performance_data['average_confidence']
        
        # Check accuracy threshold
        if accuracy < self.alert_thresholds['min_accuracy']:
            alerts.append({
                'type': 'CRITICAL',
                'message': f'Accuracy below threshold: {accuracy:.1f}% < {self.alert_thresholds["min_accuracy"]}%',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check confidence levels
        if avg_confidence < 0.5:
            alerts.append({
                'type': 'WARNING',
                'message': f'Average confidence is low: {avg_confidence:.3f}',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check for consecutive losses
        individual_results = performance_data['individual_results']
        consecutive_losses = 0
        max_consecutive = 0
        
        for result in individual_results:
            if not result['is_correct']:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        if max_consecutive >= self.alert_thresholds['max_consecutive_losses']:
            alerts.append({
                'type': 'WARNING',
                'message': f'Consecutive losses detected: {max_consecutive}',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return alerts
    
    def get_symbol_performance_breakdown(self, performance_data):
        """Get performance breakdown by symbol"""
        if not performance_data or 'individual_results' not in performance_data:
            return {}
        
        symbol_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'confidence_sum': 0})
        
        for result in performance_data['individual_results']:
            symbol = result['symbol']
            symbol_stats[symbol]['total'] += 1
            if result['is_correct']:
                symbol_stats[symbol]['correct'] += 1
            symbol_stats[symbol]['confidence_sum'] += result['confidence']
        
        # Calculate accuracy and average confidence per symbol
        breakdown = {}
        for symbol, stats in symbol_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            avg_confidence = stats['confidence_sum'] / stats['total'] if stats['total'] > 0 else 0
            
            breakdown[symbol] = {
                'accuracy': round(accuracy, 2),
                'total_decisions': stats['total'],
                'correct_decisions': stats['correct'],
                'average_confidence': round(avg_confidence, 3)
            }
        
        return breakdown
    
    def generate_monitoring_report(self, hours_back=6):
        """Generate a comprehensive monitoring report"""
        print(f"🔍 Generating AI monitoring report for last {hours_back} hours...")
        
        # Get recent performance
        decisions = self.get_recent_performance(hours_back)
        performance_data = self.calculate_real_time_accuracy(decisions)
        
        # Check for alerts
        alerts = self.check_performance_alerts(performance_data)
        
        # Get symbol breakdown
        symbol_breakdown = self.get_symbol_performance_breakdown(performance_data)
        
        # Create report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_period_hours': hours_back,
            'performance_data': performance_data,
            'alerts': alerts,
            'symbol_breakdown': symbol_breakdown,
            'total_decisions': len(decisions)
        }
        
        # Save report
        self.save_monitoring_report(report)
        
        # Print summary
        self.print_monitoring_summary(report)
        
        return report
    
    def save_monitoring_report(self, report):
        """Save monitoring report to file"""
        output_path = os.path.join(os.path.dirname(__file__), "ai_monitoring_report.json")
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📁 Monitoring report saved to {output_path}")
    
    def print_monitoring_summary(self, report):
        """Print monitoring summary"""
        print("\n" + "="*60)
        print("📊 AI SIGNALS MONITORING REPORT")
        print("="*60)
        print(f"📅 Report Time: {report['timestamp']}")
        print(f"⏰ Monitoring Period: {report['monitoring_period_hours']} hours")
        print(f"📈 Total Decisions: {report['total_decisions']}")
        
        if report['performance_data']:
            perf = report['performance_data']
            print(f"🎯 Real-time Accuracy: {perf['accuracy']:.1f}%")
            print(f"💪 Average Confidence: {perf['average_confidence']:.3f}")
            print(f"✅ Correct Predictions: {perf['correct_predictions']}/{perf['total_predictions']}")
        else:
            print("❌ No performance data available")
        
        # Print alerts
        if report['alerts']:
            print(f"\n🚨 ALERTS ({len(report['alerts'])}):")
            for alert in report['alerts']:
                print(f"   {alert['type']}: {alert['message']}")
        else:
            print("\n✅ No alerts - System performing normally")
        
        # Print symbol breakdown
        if report['symbol_breakdown']:
            print(f"\n📊 SYMBOL PERFORMANCE:")
            for symbol, stats in report['symbol_breakdown'].items():
                print(f"   {symbol}: {stats['accuracy']:.1f}% accuracy "
                      f"({stats['correct_decisions']}/{stats['total_decisions']}) "
                      f"conf: {stats['average_confidence']:.3f}")
        
        print("="*60)
    
    def start_continuous_monitoring(self, check_interval_minutes=30):
        """Start continuous monitoring with specified interval"""
        print(f"🔄 Starting continuous AI monitoring (checking every {check_interval_minutes} minutes)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Generate monitoring report
                self.generate_monitoring_report(hours_back=6)
                
                # Wait for next check
                print(f"\n⏰ Next check in {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    # Initialize monitoring system
    monitor = AIMonitoring()
    
    # Generate single report
    report = monitor.generate_monitoring_report(hours_back=6)
    
    # Uncomment to start continuous monitoring
    # monitor.start_continuous_monitoring(check_interval_minutes=30)
