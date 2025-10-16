#!/usr/bin/env python3
"""
AI Signals Historical Accuracy Test

Uses previous day's data to generate signals and checks accuracy against today's actual prices.
This provides immediate accuracy results without waiting for future data.
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_decision_engine import AIDecisionEngine
from db.database import TradeDatabase

class AIHistoricalAccuracy:
    def __init__(self):
        """Initialize the historical accuracy tester"""
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
    
    def get_previous_day_data(self, symbol: str, target_date: str) -> Optional[Dict]:
        """Get previous day's data for signal generation"""
        try:
            # Get data from the previous day
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                AND DATE(timestamp) = ?
                AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL AND volume IS NOT NULL
                AND open != 'NaN' AND high != 'NaN' AND low != 'NaN' AND close != 'NaN' AND volume != 'NaN'
                AND open != '' AND high != '' AND low != '' AND close != '' AND volume != ''
                ORDER BY timestamp DESC
            """
            
            self.cursor.execute(query, (symbol, target_date))
            data = self.cursor.fetchall()
            
            if not data:
                return None
            
            # Get the latest candle from previous day
            latest_candle = data[0]
            return {
                'open': float(latest_candle[0]),
                'high': float(latest_candle[1]),
                'low': float(latest_candle[2]),
                'close': float(latest_candle[3]),
                'volume': float(latest_candle[4]),
                'timestamp': latest_candle[5]
            }
        except Exception as e:
            print(f"❌ Error getting previous day data for {symbol}: {e}")
            return None
    
    def get_today_price(self, symbol: str, target_date: str) -> Optional[float]:
        """Get today's price for accuracy calculation"""
        try:
            # Get today's price (next day after target_date)
            next_day = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            
            query = """
                SELECT close FROM historical_data_1h 
                WHERE symbol = ? 
                AND DATE(timestamp) = ?
                AND close IS NOT NULL AND close != 'NaN' AND close != ''
                ORDER BY timestamp ASC
                LIMIT 1
            """
            
            self.cursor.execute(query, (symbol, next_day))
            result = self.cursor.fetchone()
            
            return float(result[0]) if result else None
        except Exception as e:
            print(f"❌ Error getting today's price for {symbol}: {e}")
            return None
    
    def generate_historical_signal(self, symbol: str, target_date: str) -> Optional[Dict]:
        """Generate signal using previous day's data"""
        try:
            # Get previous day's data
            prev_data = self.get_previous_day_data(symbol, target_date)
            if not prev_data:
                return None
            
            # Get historical candles for technical analysis
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                AND DATE(timestamp) <= ?
                AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL AND volume IS NOT NULL
                AND open != 'NaN' AND high != 'NaN' AND low != 'NaN' AND close != 'NaN' AND volume != 'NaN'
                AND open != '' AND high != '' AND close != '' AND volume != ''
                ORDER BY timestamp DESC
                LIMIT 24
            """
            
            self.cursor.execute(query, (symbol, target_date))
            candles = self.cursor.fetchall()
            
            if len(candles) < 5:
                return None
            
            # Convert to DataFrame format for technical analysis
            import pandas as pd
            df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate technical indicators
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df.dropna()
            
            if len(df) < 5:
                return None
            
            # Simple technical analysis
            current_price = df['close'].iloc[-1]
            sma_5 = df['close'].rolling(5).mean().iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else sma_5
            
            # Simple signal logic
            if current_price > sma_5 and current_price > sma_20:
                action = 'BUY'
                confidence = 0.8
            elif current_price < sma_5 and current_price < sma_20:
                action = 'SELL'
                confidence = 0.8
            else:
                action = 'HOLD'
                confidence = 0.6
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'current_price': current_price,
                'sma_5': sma_5,
                'sma_20': sma_20,
                'target_date': target_date
            }
            
        except Exception as e:
            print(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    def calculate_accuracy(self, signals: List[Dict], target_date: str) -> Dict:
        """Calculate accuracy of signals against today's prices"""
        total_signals = len(signals)
        correct_predictions = 0
        evaluated_signals = 0
        results = []
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            predicted_price = signal['current_price']
            
            # Get today's actual price
            actual_price = self.get_today_price(symbol, target_date)
            
            if actual_price is None:
                results.append({
                    'symbol': symbol,
                    'action': action,
                    'predicted_price': predicted_price,
                    'actual_price': 'N/A',
                    'price_change': 'N/A',
                    'correct': False,
                    'status': 'No data'
                })
                continue
            
            evaluated_signals += 1
            price_change = (actual_price - predicted_price) / predicted_price
            
            # Determine if prediction was correct
            is_correct = False
            if action == 'BUY' and price_change > 0.001:  # 0.1% threshold
                is_correct = True
            elif action == 'SELL' and price_change < -0.001:
                is_correct = True
            elif action == 'HOLD' and abs(price_change) <= 0.001:
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'symbol': symbol,
                'action': action,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'price_change': price_change,
                'correct': is_correct,
                'status': 'Evaluated'
            })
        
        accuracy = (correct_predictions / evaluated_signals * 100) if evaluated_signals > 0 else 0
        
        return {
            'total_signals': total_signals,
            'evaluated_signals': evaluated_signals,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'results': results
        }
    
    def run_historical_accuracy_test(self, target_date: str = None):
        """Run the historical accuracy test"""
        if target_date is None:
            # Use yesterday's date
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print("🧪 AI SIGNALS HISTORICAL ACCURACY TEST")
        print("=" * 60)
        print(f"📅 Using data from: {target_date}")
        print(f"🎯 Checking accuracy against: {(datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        if not self.connect_db():
            return False
        
        try:
            # Get available symbols
            self.cursor.execute("SELECT DISTINCT symbol FROM historical_data_1h WHERE symbol LIKE '%{=h}'")
            symbols = [row[0] for row in self.cursor.fetchall()]
            
            if not symbols:
                print("❌ No symbols found in database")
                return False
            
            print(f"📊 Found {len(symbols)} symbols to test")
            print("-" * 60)
            
            # Generate signals for each symbol
            signals = []
            for symbol in symbols:
                print(f"🤖 Generating signal for {symbol}...")
                signal = self.generate_historical_signal(symbol, target_date)
                if signal:
                    signals.append(signal)
                    print(f"   ✅ {signal['action']} (confidence: {signal['confidence']:.2f})")
                else:
                    print(f"   ❌ No data available")
            
            if not signals:
                print("❌ No signals generated")
                return False
            
            print(f"\n📈 Generated {len(signals)} signals")
            print("-" * 60)
            
            # Calculate accuracy
            print("🎯 Calculating accuracy...")
            accuracy_results = self.calculate_accuracy(signals, target_date)
            
            # Display results
            self.print_accuracy_results(accuracy_results, target_date)
            
            return True
            
        finally:
            self.close_db()
    
    def print_accuracy_results(self, results: Dict, target_date: str):
        """Print formatted accuracy results"""
        print("\n📊 HISTORICAL ACCURACY RESULTS")
        print("=" * 60)
        print(f"📅 Test Date: {target_date}")
        print(f"🎯 Check Date: {(datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        print(f"📈 Total Signals: {results['total_signals']}")
        print(f"✅ Evaluated Signals: {results['evaluated_signals']}")
        print(f"🎯 Correct Predictions: {results['correct_predictions']}")
        print(f"📊 Accuracy Rate: {results['accuracy']:.1f}%")
        
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
        
        # Individual results
        print(f"\n📋 INDIVIDUAL RESULTS:")
        print("-" * 60)
        print(f"{'Symbol':<20} {'Action':<8} {'Predicted':<12} {'Actual':<12} {'Change':<10} {'Status':<10}")
        print("-" * 60)
        
        for result in results['results']:
            change_str = f"{result['price_change']:.3f}" if result['price_change'] != 'N/A' else 'N/A'
            status_emoji = "✅" if result['correct'] else "❌"
            actual_price_str = f"${result['actual_price']:.2f}" if result['actual_price'] != 'N/A' else 'N/A'
            print(f"{result['symbol']:<20} {result['action']:<8} ${result['predicted_price']:<11.2f} {actual_price_str:<12} {change_str:<10} {status_emoji} {result['status']}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Signals Historical Accuracy Test')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD) to use for signal generation')
    
    args = parser.parse_args()
    
    tester = AIHistoricalAccuracy()
    
    if args.date:
        success = tester.run_historical_accuracy_test(args.date)
    else:
        success = tester.run_historical_accuracy_test()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
