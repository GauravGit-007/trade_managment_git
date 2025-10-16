# Check AI decisions in database
import sqlite3
import os
import sys

# Add parent directory to path
sys.path.append('..')
from db.database import TradeDatabase

def check_ai_decisions():
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    try:
        # Check if ai_decisions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_decisions'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Count total decisions
            cursor.execute('SELECT COUNT(*) FROM ai_decisions')
            total_decisions = cursor.fetchone()[0]
            
            # Count recent decisions
            cursor.execute("SELECT COUNT(*) FROM ai_decisions WHERE datetime(decision_timestamp) >= datetime('now', '-24 hours')")
            recent_decisions = cursor.fetchone()[0]
            
            print(f'📊 AI DECISIONS IN DATABASE:')
            print(f'   Total decisions: {total_decisions}')
            print(f'   Last 24 hours: {recent_decisions}')
            
            # Show sample decisions
            cursor.execute('SELECT symbol, action, confidence, decision_timestamp FROM ai_decisions ORDER BY decision_timestamp DESC LIMIT 5')
            decisions = cursor.fetchall()
            
            print(f'\n📋 RECENT DECISIONS:')
            for decision in decisions:
                print(f'   {decision[0]}: {decision[1]} (conf: {decision[2]}) at {decision[3]}')
                
            # Check accuracy data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_evaluations'")
            eval_table_exists = cursor.fetchone()
            
            if eval_table_exists:
                cursor.execute('SELECT COUNT(*) FROM ai_evaluations')
                total_evaluations = cursor.fetchone()[0]
                print(f'\n📈 EVALUATION DATA:')
                print(f'   Total evaluations: {total_evaluations}')
            else:
                print(f'\n📈 EVALUATION DATA: No evaluations yet (need 4+ hours of data)')
                
        else:
            print('❌ ai_decisions table not found - no AI decisions recorded yet')
            print('   Run: python ai_workflow.py --mode signals')
            
    except Exception as e:
        print(f'❌ Error checking database: {e}')
    finally:
        db.close_connection(conn)

if __name__ == "__main__":
    check_ai_decisions()


