#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

def check_database_structure():
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print("Available tables:", tables)
    
    # Check historical data table structure
    if 'historical_data_1h' in tables:
        cursor.execute("PRAGMA table_info(historical_data_1h)")
        columns = cursor.fetchall()
        print("\nhistorical_data_1h columns:", columns)
        
        # Get a sample row
        cursor.execute("SELECT * FROM historical_data_1h LIMIT 1")
        sample = cursor.fetchone()
        print("Sample row:", sample)
    
    # Check if there are other tables with timestamp data
    for table in tables:
        if 'historical' in table.lower() or 'data' in table.lower():
            print(f"\nChecking {table}:")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print(f"  Columns: {columns}")
    
    db.close_connection(conn)

if __name__ == "__main__":
    check_database_structure()
