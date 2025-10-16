import os
import sys
from datetime import datetime, timezone

# Ensure root on sys.path to import db.database
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
root_dir = os.path.dirname(grandparent_dir)
sys.path.append(root_dir)

from db.database import TradeDatabase


def main() -> None:
    conn, cursor = TradeDatabase.sql_connect()

    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Smart_AI_decisions'")
    exists = cursor.fetchone() is not None
    print(f"Table exists: {exists}")
    if not exists:
        TradeDatabase.close_connection(conn)
        return

    # Inspect schema
    cursor.execute("PRAGMA table_info(Smart_AI_decisions)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Columns: {cols}")

    # Show recent rows (all columns)
    cursor.execute(
        """
        SELECT *
        FROM Smart_AI_decisions
        ORDER BY rowid DESC
        LIMIT 16
        """
    )
    recent = cursor.fetchall()
    print(f"Recent rows: {len(recent)}")
    for r in recent:
        print("  ", r)

    # Today rows
    today = datetime.now(timezone.utc).date().isoformat()
    # Try common timestamp/date columns
    date_filters = [
        ("timestamp", "substr(timestamp, 1, 10) = ?"),
        ("decision_timestamp", "substr(decision_timestamp, 1, 10) = ?"),
        ("created_at", "substr(created_at, 1, 10) = ?"),
    ]
    today_rows = []
    used_filter = None
    for col, where_clause in date_filters:
        if col in cols:
            try:
                cursor.execute(
                    f"SELECT * FROM Smart_AI_decisions WHERE {where_clause} ORDER BY rowid DESC",
                    (today,),
                )
                today_rows = cursor.fetchall()
                used_filter = col
                break
            except Exception:
                continue
    today_rows = cursor.fetchall()
    print(f"\nToday's rows: {len(today_rows)} ({today}) using column: {used_filter}")
    for r in today_rows:
        print("  ", r)

    TradeDatabase.close_connection(conn)


if __name__ == "__main__":
    main()


