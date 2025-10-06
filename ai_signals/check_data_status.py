# Check Data Status - Verify if we have enough data for accuracy calculation

import sqlite3
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append('..')
from db.database import TradeDatabase

def check_data_status():
    """Check if we have enough data for accuracy calculation"""
    print("🔍 DATA STATUS CHECK")
    print("=" * 40)
    
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    try:
        # Check AI decisions
        cursor.execute("SELECT COUNT(*) FROM ai_decisions")
        ai_count = cursor.fetchone()[0]
        print(f"📊 AI Decisions: {ai_count}")
        
        # Check historical data
        cursor.execute("SELECT COUNT(*) FROM historical_data_1h")
        hist_count = cursor.fetchone()[0]
        print(f"📈 Historical Data Points: {hist_count}")
        
        # Check data by symbol
        cursor.execute("""
            SELECT symbol, COUNT(*) as count, 
                   MIN(timestamp) as earliest, 
                   MAX(timestamp) as latest
            FROM historical_data_1h 
            GROUP BY symbol 
            ORDER BY symbol
        """)
        symbol_data = cursor.fetchall()
        
        print(f"\n📊 DATA BY SYMBOL:")
        for symbol, count, earliest, latest in symbol_data:
            print(f"   {symbol}: {count} points ({earliest} to {latest})")
        
        # Check if we have recent data
        cursor.execute("""
            SELECT MAX(timestamp) as latest_data
            FROM historical_data_1h
        """)
        latest_data = cursor.fetchone()[0]
        
        if latest_data:
            latest_time = datetime.fromisoformat(latest_data.replace('Z', '+00:00'))
            hours_ago = (datetime.utcnow() - latest_time).total_seconds() / 3600
            print(f"\n⏰ LATEST DATA:")
            print(f"   Most recent: {latest_data}")
            print(f"   Hours ago: {hours_ago:.1f}")
            
            if hours_ago > 24:
                print(f"   ⚠️  Data is {hours_ago:.1f} hours old - may need updating")
            else:
                print(f"   ✅ Data is recent")
        
        # Check specific symbol data
        print(f"\n🔍 DETAILED CHECK FOR /ES:XCME:")
        cursor.execute("""
            SELECT COUNT(*) as count,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM historical_data_1h 
            WHERE symbol = '/ES:XCME{=h}'
        """)
        es_data = cursor.fetchone()
        
        if es_data[0] > 0:
            print(f"   Data points: {es_data[0]}")
            print(f"   Range: {es_data[1]} to {es_data[2]}")
            
            # Check if we have data after the oldest AI decision
            cursor.execute("""
                SELECT MIN(decision_timestamp) 
                FROM ai_decisions
            """)
            oldest_decision = cursor.fetchone()[0]
            
            if oldest_decision:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM historical_data_1h 
                    WHERE symbol = '/ES:XCME{=h}'
                    AND datetime(timestamp) > datetime(?)
                """, (oldest_decision,))
                future_count = cursor.fetchone()[0]
                
                print(f"   Data after oldest decision: {future_count} points")
                
                if future_count > 0:
                    print(f"   ✅ Can evaluate AI decisions!")
                else:
                    print(f"   ❌ No data after AI decisions - need to update historical data")
                    print(f"   💡 Run: cd .. && python services/historical_data.py")
        else:
            print(f"   ❌ No data found for /ES:XCME")
            print(f"   💡 Run: cd .. && python services/historical_data.py")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if hist_count < 100:
            print(f"   📥 Update historical data: cd .. && python services/historical_data.py")
        if ai_count > 0 and hist_count > 100:
            print(f"   ✅ Ready for accuracy calculation!")
            print(f"   🚀 Run: python accuracy_calculator.py --hours 48")
        else:
            print(f"   ⏳ Need more data before accuracy calculation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        db.close_connection(conn)

if __name__ == "__main__":
    check_data_status()
