import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase


def parse_iso_utc(ts: str) -> datetime:
    try:
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fetch_decisions(conn: sqlite3.Connection, symbol: Optional[str], start: Optional[datetime], end: Optional[datetime]):
    cur = conn.cursor()
    where = []
    params: List[object] = []
    if symbol:
        where.append("symbol = ?")
        params.append(symbol)
    if start:
        where.append("datetime(decision_timestamp) >= datetime(?)")
        params.append(start.isoformat())
    if end:
        where.append("datetime(decision_timestamp) <= datetime(?)")
        params.append(end.isoformat())
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT id, symbol, decision_timestamp, action
        FROM rl_decisions
        {where_sql}
        ORDER BY decision_timestamp
    """
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    return rows


def fetch_first_bar_at_or_after(cur: sqlite3.Cursor, symbol: str, ts: datetime) -> Optional[tuple]:
    cur.execute(
        """
        SELECT timestamp, close FROM historical_data_1h
        WHERE symbol = ? AND datetime(timestamp) >= datetime(?)
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (symbol, ts.isoformat()),
    )
    row = cur.fetchone()
    return row


def evaluate(
    symbol: Optional[str],
    horizon_hours: int,
    start: Optional[datetime],
    end: Optional[datetime],
):
    conn, cursor = TradeDatabase.sql_connect()
    if conn is None:
        raise RuntimeError("Could not connect to SQLite database.")

    decisions = fetch_decisions(conn, symbol, start, end)
    if not decisions:
        print("No decisions found for the given filters.")
        TradeDatabase.close_connection(conn)
        return

    total = 0
    total_traded = 0
    evaluable = 0
    hits_all = 0
    hits_traded = 0
    sum_pnl_delta = 0.0  # executed_delta * price_diff

    for dec in decisions:
        dec_id, sym, dec_ts_str, action_str = dec
        try:
            dec_ts = parse_iso_utc(dec_ts_str)
        except Exception:
            continue

        # Determine direction from action (fallback to experiences for executed delta)
        try:
            a_int = int(action_str)
        except Exception:
            a_int = 0
        # Map action index to delta units as in env
        action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        executed_delta = float(action_map.get(a_int, 0))

        # Find start/end bars
        start_bar = fetch_first_bar_at_or_after(cursor, sym, dec_ts)
        if not start_bar:
            total += 1
            continue
        start_ts_str, price_start = start_bar
        try:
            start_ts = parse_iso_utc(start_ts_str)
        except Exception:
            total += 1
            continue

        end_target = start_ts + timedelta(hours=horizon_hours)
        end_bar = fetch_first_bar_at_or_after(cursor, sym, end_target)
        if not end_bar:
            total += 1
            continue
        end_ts_str, price_end = end_bar

        try:
            price_start_f = float(price_start)
            price_end_f = float(price_end)
        except Exception:
            total += 1
            continue

        price_diff = price_end_f - price_start_f
        direction = 0
        if executed_delta > 0:
            direction = 1
        elif executed_delta < 0:
            direction = -1

        total += 1
        if executed_delta != 0:
            total_traded += 1

        if price_diff == 0:
            evaluable += 1
            # Neutral price move; do not count as hit
        else:
            evaluable += 1
            outcome = 0
            if (direction > 0 and price_diff > 0) or (direction < 0 and price_diff < 0):
                outcome = 1
            # Count hits
            if outcome == 1:
                hits_all += 1
                if executed_delta != 0:
                    hits_traded += 1

        # Simple delta PnL proxy over horizon
        sum_pnl_delta += executed_delta * price_diff

    TradeDatabase.close_connection(conn)

    acc_all = (hits_all / evaluable * 100.0) if evaluable > 0 else 0.0
    acc_traded = (hits_traded / total_traded * 100.0) if total_traded > 0 else 0.0

    print("Evaluation summary")
    print(f"- Horizon: {horizon_hours}h")
    if symbol:
        print(f"- Symbol: {symbol}")
    print(f"- Decisions total: {total}")
    print(f"- Decisions evaluable: {evaluable}")
    print(f"- Decisions with trades (non-hold): {total_traded}")
    print(f"- Directional accuracy (all evaluable): {acc_all:.2f}%")
    print(f"- Directional accuracy (traded only): {acc_traded:.2f}%")
    print(f"- Sum(delta * price_diff) over horizon: {sum_pnl_delta:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL decisions for hit-rate and PnL proxy.")
    parser.add_argument("--symbol", type=str, default=None, help="Filter by symbol (exact match).")
    parser.add_argument("--horizon_hours", type=int, default=1, help="Horizon in hours for evaluation (default: 1).")
    parser.add_argument("--start", type=str, default=None, help="ISO start datetime filter.")
    parser.add_argument("--end", type=str, default=None, help="ISO end datetime filter.")
    args = parser.parse_args()

    start_dt = parse_iso_utc(args.start) if args.start else None
    end_dt = parse_iso_utc(args.end) if args.end else None

    evaluate(
        symbol=args.symbol,
        horizon_hours=args.horizon_hours,
        start=start_dt,
        end=end_dt,
    )


if __name__ == "__main__":
    main()

