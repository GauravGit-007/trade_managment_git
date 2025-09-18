import os
import sys
import sqlite3
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase


def parse_iso(ts: str) -> datetime:
    # Support both Z and +00:00 formats
    try:
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        # Last resort: try without timezone
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def iso_variants(dt: datetime) -> tuple[str, str]:
    z = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plus = dt.astimezone(timezone.utc).isoformat()
    return z, plus


def backfill_actuals() -> int:
    conn, cursor = TradeDatabase.sql_connect()
    if conn is None:
        raise RuntimeError("Could not connect to SQLite database.")

    # Get predictions where target time has passed and actual is NULL
    cursor.execute(
        """
        SELECT id, symbol, target_timestamp, predicted_value
        FROM lstm_predictions
        WHERE actual_value IS NULL
        AND datetime(target_timestamp) <= datetime('now')
        """
    )
    rows = cursor.fetchall()

    updated = 0
    for pred_id, symbol, target_ts, pred_val in rows:
        try:
            target_dt = parse_iso(target_ts)
            z, plus = iso_variants(target_dt)
            # Try match both variants
            cursor.execute(
                """
                SELECT close FROM historical_data_1h
                WHERE symbol = ? AND (timestamp = ? OR timestamp = ?)
                LIMIT 1
                """,
                (symbol, z, plus),
            )
            row = cursor.fetchone()
            if row is None:
                # Looser match: same hour
                hour_prefix = target_dt.strftime("%Y-%m-%dT%H:")
                cursor.execute(
                    """
                    SELECT close FROM historical_data_1h
                    WHERE symbol = ? AND substr(timestamp, 1, 14) = ?
                    ORDER BY timestamp LIMIT 1
                    """,
                    (symbol, hour_prefix),
                )
                row = cursor.fetchone()

            if row is not None:
                actual = float(row[0])
                cursor.execute(
                    "UPDATE lstm_predictions SET actual_value = ? WHERE id = ?",
                    (actual, pred_id),
                )
                updated += 1
        except Exception:
            continue

    conn.commit()
    TradeDatabase.close_connection(conn)
    return updated


if __name__ == "__main__":
    count = backfill_actuals()
    print(f"Updated actuals for {count} prediction rows.")

