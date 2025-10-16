#!/usr/bin/env python3
"""
Script to empty the rl_experiences table in the trade management database.
This helps prevent disk I/O errors caused by the table growing too large.

Usage: python empty_rl_experiences.py
"""

import sqlite3
import os
import sys
from datetime import datetime
from db.database import TradeDatabase

def empty_rl_experiences_table():
    """
    Empty the rl_experiences table and provide feedback on the operation.
    """
    try:
        # Get database path
        db_path = TradeDatabase.get_db_path()
        print(f"Database path: {db_path}")
        
        # Check if database file exists
        if not os.path.exists(db_path):
            print(f"❌ Error: Database file not found at {db_path}")
            return False
        
        # Connect to database
        print("Connecting to database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if rl_experiences table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_experiences'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("⚠️  Warning: rl_experiences table does not exist")
            conn.close()
            return True
        
        # Get current row count
        cursor.execute("SELECT COUNT(*) FROM rl_experiences")
        current_count = cursor.fetchone()[0]
        print(f"Current rl_experiences table has {current_count:,} rows")
        
        if current_count == 0:
            print("✅ rl_experiences table is already empty")
            conn.close()
            return True
        
        # Confirm before deletion (optional - can be removed for automation)
        print(f"\n⚠️  About to delete {current_count:,} rows from rl_experiences table")
        print("This action cannot be undone!")
        
        # For automation, you can comment out the next 3 lines
        response = input("Do you want to continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Operation cancelled by user")
            conn.close()
            return False
        
        # Create backup of current data (optional)
        backup_choice = input("Create backup before deletion? (y/N): ").strip().lower()
        if backup_choice in ['y', 'yes']:
            backup_path = f"rl_experiences_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            print(f"Creating backup at: {backup_path}")
            
            # Create a backup database with just the rl_experiences table
            backup_conn = sqlite3.connect(backup_path)
            backup_cursor = backup_conn.cursor()
            
            # Get table structure
            cursor.execute("PRAGMA table_info(rl_experiences)")
            columns = cursor.fetchall()
            
            # Create table in backup
            create_sql = f"""
                CREATE TABLE rl_experiences (
                    {', '.join([f"{col[1]} {col[2]}" + (" PRIMARY KEY" if col[5] else "") for col in columns])}
                )
            """
            backup_cursor.execute(create_sql)
            
            # Copy data
            cursor.execute("SELECT * FROM rl_experiences")
            data = cursor.fetchall()
            
            if data:
                placeholders = ','.join(['?' for _ in columns])
                backup_cursor.executemany(f"INSERT INTO rl_experiences VALUES ({placeholders})", data)
                backup_conn.commit()
                print(f"✅ Backup created with {len(data):,} rows")
            
            backup_conn.close()
        
        # Empty the table
        print("Emptying rl_experiences table...")
        start_time = datetime.now()
        
        cursor.execute("DELETE FROM rl_experiences")
        conn.commit()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM rl_experiences")
        new_count = cursor.fetchone()[0]
        
        print(f"✅ Successfully emptied rl_experiences table")
        print(f"📊 Rows deleted: {current_count:,}")
        print(f"📊 Remaining rows: {new_count:,}")
        print(f"⏱️  Operation took: {duration:.2f} seconds")
        
        # Test database write operations
        print("\nTesting database write operations...")
        try:
            cursor.execute("CREATE TABLE IF NOT EXISTS test_write (id INTEGER PRIMARY KEY, val TEXT)")
            cursor.execute("INSERT INTO test_write (val) VALUES (?)", ("test",))
            conn.commit()
            
            cursor.execute("SELECT * FROM test_write")
            result = cursor.fetchall()
            print(f"✅ Write test successful: {result}")
            
            cursor.execute("DROP TABLE test_write")
            conn.commit()
            print("✅ Database write operations are working correctly")
            
        except Exception as e:
            print(f"❌ Write test failed: {e}")
        
        conn.close()
        return True
        
    except sqlite3.OperationalError as e:
        print(f"❌ SQLite Operational Error: {e}")
        print("This might indicate database corruption or disk I/O issues")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the rl_experiences table cleanup.
    """
    print("=" * 60)
    print("RL_EXPERIENCES TABLE CLEANUP SCRIPT")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = empty_rl_experiences_table()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SCRIPT COMPLETED SUCCESSFULLY")
    else:
        print("❌ SCRIPT COMPLETED WITH ERRORS")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
