# Quick Accuracy Check - Shows current status and when accuracy will be available

import sqlite3
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append('..')
from db.database import TradeDatabase

def quick_accuracy_check():
    """Quick check of current accuracy status"""
    print("🔍 AI SIGNALS ACCURACY STATUS CHECK")
    print("=" * 50)
    
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    try:
        # Check AI decisions
        cursor.execute("SELECT COUNT(*) FROM ai_decisions")
        total_decisions = cursor.fetchone()[0]
        
        # Get recent decisions
        cursor.execute("""
            SELECT symbol, action, confidence, decision_timestamp 
            FROM ai_decisions 
            ORDER BY decision_timestamp DESC 
            LIMIT 5
        """)
        recent_decisions = cursor.fetchall()
        
        print(f"📊 AI DECISIONS IN DATABASE:")
        print(f"   Total decisions: {total_decisions}")
        print(f"   Recent decisions:")
        
        for decision in recent_decisions:
            symbol, action, confidence, timestamp = decision
            print(f"     {symbol}: {action} (conf: {confidence}) at {timestamp}")
        
        # Check oldest decision
        cursor.execute("""
            SELECT MIN(decision_timestamp) as oldest_decision
            FROM ai_decisions
        """)
        oldest_decision = cursor.fetchone()[0]
        
        if oldest_decision:
            oldest_time = datetime.fromisoformat(oldest_decision.replace('Z', '+00:00'))
            hours_ago = (datetime.utcnow() - oldest_time).total_seconds() / 3600
            
            print(f"\n⏰ TIMING ANALYSIS:")
            print(f"   Oldest decision: {oldest_decision}")
            print(f"   Hours ago: {hours_ago:.1f}")
            
            if hours_ago >= 4:
                print(f"   ✅ Ready for evaluation! (4+ hours old)")
                
                # Try to evaluate the oldest decision
                cursor.execute("""
                    SELECT symbol, action, current_price, decision_timestamp
                    FROM ai_decisions 
                    WHERE decision_timestamp = ?
                """, (oldest_decision,))
                oldest_decision_data = cursor.fetchone()
                
                if oldest_decision_data:
                    symbol, action, price, timestamp = oldest_decision_data
                    print(f"\n🔍 EVALUATING OLDEST DECISION:")
                    print(f"   Symbol: {symbol}")
                    print(f"   Action: {action}")
                    print(f"   Price: ${price}")
                    print(f"   Time: {timestamp}")
                    
                    # Check if we have price data after this decision
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM historical_data_1h 
                        WHERE symbol = ? 
                        AND datetime(timestamp) > datetime(?)
                    """, (symbol, timestamp))
                    future_data_count = cursor.fetchone()[0]
                    
                    print(f"   Future price data points: {future_data_count}")
                    
                    if future_data_count > 0:
                        print(f"   ✅ Can evaluate this decision!")
                        
                        # Get the price data
                        cursor.execute("""
                            SELECT close, timestamp
                            FROM historical_data_1h 
                            WHERE symbol = ? 
                            AND datetime(timestamp) > datetime(?)
                            ORDER BY timestamp ASC
                            LIMIT 4
                        """, (symbol, timestamp))
                        price_data = cursor.fetchall()
                        
                        if price_data:
                            entry_price = price
                            final_price = price_data[-1][0]
                            price_change = (final_price - entry_price) / entry_price
                            
                            print(f"   Entry price: ${entry_price}")
                            print(f"   Final price: ${final_price}")
                            print(f"   Price change: {price_change:.4f} ({price_change*100:.2f}%)")
                            
                            # Check if correct
                            is_correct = False
                            if action in ['STRONG_BUY', 'BUY']:
                                is_correct = price_change > 0.001
                            elif action in ['STRONG_SELL', 'SELL']:
                                is_correct = price_change < -0.001
                            elif action == 'HOLD':
                                is_correct = abs(price_change) <= 0.001
                            
                            result = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                            print(f"   Result: {result}")
                            
                            if is_correct:
                                print(f"   🎉 AI prediction was accurate!")
                            else:
                                print(f"   📉 AI prediction was not accurate")
                    else:
                        print(f"   ❌ No future price data available for evaluation")
            else:
                hours_needed = 4 - hours_ago
                print(f"   ⏳ Need {hours_needed:.1f} more hours for evaluation")
                print(f"   📅 Will be ready at: {(datetime.utcnow() + timedelta(hours=hours_needed)).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Check if we have any evaluations
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_evaluations'")
        eval_table_exists = cursor.fetchone()
        
        if eval_table_exists:
            cursor.execute("SELECT COUNT(*) FROM ai_evaluations")
            total_evaluations = cursor.fetchone()[0]
            print(f"\n📈 EVALUATION DATA:")
            print(f"   Total evaluations: {total_evaluations}")
        else:
            print(f"\n📈 EVALUATION DATA:")
            print(f"   No evaluation table found - evaluations will be created automatically")
        
        print(f"\n🎯 NEXT STEPS:")
        if hours_ago >= 4:
            print(f"   ✅ Run: python accuracy_calculator.py --hours 48")
            print(f"   ✅ Run: python ai_workflow.py --mode evaluate --hours 24")
        else:
            print(f"   ⏳ Wait {4-hours_ago:.1f} more hours for first accuracy results")
            print(f"   🔄 Keep continuous monitoring running")
            print(f"   📊 Check back later with: python accuracy_calculator.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        db.close_connection(conn)

if __name__ == "__main__":
    quick_accuracy_check()


