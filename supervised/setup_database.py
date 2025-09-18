"""
Database setup for supervised learning components.
Creates necessary tables for SL decisions, alerts, and monitoring.
"""

import os
import sys
import sqlite3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase


def create_sl_tables():
    """Create all necessary tables for supervised learning."""
    conn, cursor = TradeDatabase.sql_connect()
    
    # SL Decisions table (mirrors rl_decisions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sl_decisions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            decision_timestamp TEXT NOT NULL,
            state_json TEXT,
            action INTEGER NOT NULL,
            position_before REAL,
            position_after REAL,
            price REAL,
            pnl_change REAL,
            model_version TEXT,
            confidence REAL,
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # SL Alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sl_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            alert_timestamp TEXT NOT NULL,
            value REAL,
            threshold REAL,
            model_version TEXT,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # SL Model Performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sl_model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            symbol TEXT NOT NULL,
            evaluation_timestamp TEXT NOT NULL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            num_trades INTEGER,
            avg_confidence REAL,
            evaluation_period_hours INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # SL Training History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sl_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            training_timestamp TEXT NOT NULL,
            symbols TEXT,  -- JSON array of symbols
            features_count INTEGER,
            training_samples INTEGER,
            validation_accuracy REAL,
            test_accuracy REAL,
            config_json TEXT,  -- Training configuration
            model_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_decisions_symbol_timestamp ON sl_decisions(symbol, decision_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_decisions_timestamp ON sl_decisions(decision_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_alerts_timestamp ON sl_alerts(alert_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_alerts_type ON sl_alerts(alert_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_performance_model_symbol ON sl_model_performance(model_version, symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sl_training_model_type ON sl_training_history(model_type, training_timestamp)")
    
    conn.commit()
    TradeDatabase.close_connection(conn)
    
    print("Created SL database tables:")
    print("- sl_decisions: Trading decisions made by SL models")
    print("- sl_alerts: Performance alerts and notifications")
    print("- sl_model_performance: Model evaluation metrics")
    print("- sl_training_history: Training run history")


def create_sl_views():
    """Create useful views for SL data analysis."""
    conn, cursor = TradeDatabase.sql_connect()
    
    # Recent SL decisions view
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS v_sl_recent_decisions AS
        SELECT 
            symbol,
            decision_timestamp,
            action,
            position_before,
            position_after,
            price,
            pnl_change,
            confidence,
            model_version
        FROM sl_decisions
        WHERE decision_timestamp >= datetime('now', '-24 hours')
        ORDER BY decision_timestamp DESC
    """)
    
    # SL performance summary view
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS v_sl_performance_summary AS
        SELECT 
            model_version,
            symbol,
            COUNT(*) as total_decisions,
            SUM(pnl_change) as total_pnl,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN pnl_change > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate,
            MAX(decision_timestamp) as last_decision
        FROM sl_decisions
        WHERE decision_timestamp >= datetime('now', '-7 days')
        GROUP BY model_version, symbol
    """)
    
    # SL alerts summary view
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS v_sl_alerts_summary AS
        SELECT 
            alert_type,
            severity,
            COUNT(*) as alert_count,
            MAX(alert_timestamp) as last_alert
        FROM sl_alerts
        WHERE alert_timestamp >= datetime('now', '-24 hours')
        GROUP BY alert_type, severity
    """)
    
    conn.commit()
    TradeDatabase.close_connection(conn)
    
    print("Created SL database views:")
    print("- v_sl_recent_decisions: Recent trading decisions")
    print("- v_sl_performance_summary: Performance metrics by model/symbol")
    print("- v_sl_alerts_summary: Recent alerts summary")


def verify_sl_tables():
    """Verify that all SL tables exist and are properly structured."""
    conn, cursor = TradeDatabase.sql_connect()
    
    # Check if tables exist
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'sl_%'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['sl_decisions', 'sl_alerts', 'sl_model_performance', 'sl_training_history']
    missing_tables = [t for t in expected_tables if t not in tables]
    
    if missing_tables:
        print(f"Missing tables: {missing_tables}")
        return False
    
    # Check table structures
    for table in expected_tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        print(f"\n{table} columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
    
    TradeDatabase.close_connection(conn)
    print("\nAll SL tables verified successfully!")
    return True


def cleanup_old_data(days_to_keep: int = 30):
    """Clean up old SL data to keep database size manageable."""
    conn, cursor = TradeDatabase.sql_connect()
    
    cutoff_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Clean old decisions
    cursor.execute("""
        DELETE FROM sl_decisions 
        WHERE decision_timestamp < datetime('now', '-{} days')
    """.format(days_to_keep))
    deleted_decisions = cursor.rowcount
    
    # Clean old alerts
    cursor.execute("""
        DELETE FROM sl_alerts 
        WHERE alert_timestamp < datetime('now', '-{} days')
    """.format(days_to_keep))
    deleted_alerts = cursor.rowcount
    
    # Clean old performance records
    cursor.execute("""
        DELETE FROM sl_model_performance 
        WHERE evaluation_timestamp < datetime('now', '-{} days')
    """.format(days_to_keep))
    deleted_performance = cursor.rowcount
    
    conn.commit()
    TradeDatabase.close_connection(conn)
    
    print(f"Cleaned up old SL data:")
    print(f"- Deleted {deleted_decisions} old decisions")
    print(f"- Deleted {deleted_alerts} old alerts")
    print(f"- Deleted {deleted_performance} old performance records")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup SL database tables")
    parser.add_argument("--create", action="store_true", help="Create SL tables")
    parser.add_argument("--views", action="store_true", help="Create SL views")
    parser.add_argument("--verify", action="store_true", help="Verify SL tables")
    parser.add_argument("--cleanup", type=int, help="Clean up data older than N days")
    parser.add_argument("--all", action="store_true", help="Run all setup operations")
    
    args = parser.parse_args()
    
    if args.all or args.create:
        create_sl_tables()
    
    if args.all or args.views:
        create_sl_views()
    
    if args.all or args.verify:
        verify_sl_tables()
    
    if args.cleanup:
        cleanup_old_data(args.cleanup)
    
    if not any([args.create, args.views, args.verify, args.cleanup, args.all]):
        print("No operation specified. Use --help for options.")
        print("Recommended: python supervised/setup_database.py --all")


if __name__ == "__main__":
    main()
