#!/usr/bin/env python3
"""
SL Agent Evaluation Script

Evaluates the accuracy and performance of decisions made by the SL (supervised learning) model.
Mirrors rl/evaluate_agent.py but sources data from sl_decisions.
"""

import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase


def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime object"""
    try:
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fetch_decisions(conn: sqlite3.Connection, symbol: Optional[str], start: Optional[datetime], end: Optional[datetime]):
    """Fetch SL decisions for evaluation"""
    cur = conn.cursor()
    where = []
    params = []

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
        SELECT id, symbol, decision_timestamp, action, position_before,
               position_after, price, pnl_change, model_version
        FROM sl_decisions
        {where_sql}
        ORDER BY decision_timestamp
    """

    cur.execute(sql, tuple(params))
    return cur.fetchall()


def fetch_first_bar_at_or_after(cur: sqlite3.Cursor, symbol: str, ts: datetime) -> Optional[tuple]:
    """Fetch the first bar at or after a given timestamp"""
    cur.execute(
        """
        SELECT timestamp, close FROM historical_data_1h
        WHERE symbol = ? AND datetime(timestamp) >= datetime(?)
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (symbol, ts.isoformat()),
    )
    return cur.fetchone()


def calculate_directional_accuracy(decisions: List[Tuple], horizon_hours: int, conn: sqlite3.Connection):
    """Calculate directional accuracy of decisions"""
    cursor = conn.cursor()
    total_decisions = len(decisions)
    correct_predictions = 0
    total_trades = 0
    correct_trades = 0

    # Action mapping from environment (0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy)
    action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}

    for decision in decisions:
        _, symbol, dec_ts_str, action_str, _, _, _, _, _ = decision

        try:
            dec_ts = parse_iso_utc(dec_ts_str)
            action = int(action_str)
            executed_delta = float(action_map.get(action, 0))
        except (ValueError, TypeError):
            continue

        # Skip holds for directional accuracy
        if executed_delta == 0:
            continue

        total_trades += 1

        # Find price at decision time and after horizon
        start_bar = fetch_first_bar_at_or_after(cursor, symbol, dec_ts)
        if not start_bar:
            continue

        start_ts_str, price_start = start_bar
        try:
            start_ts = parse_iso_utc(start_ts_str)
            end_target = start_ts + timedelta(hours=horizon_hours)
            end_bar = fetch_first_bar_at_or_after(cursor, symbol, end_target)

            if not end_bar:
                continue

            _, price_end = end_bar
            price_start_f = float(price_start)
            price_end_f = float(price_end)

            price_diff = price_end_f - price_start_f

            # Determine if prediction was correct
            if (executed_delta > 0 and price_diff > 0) or (executed_delta < 0 and price_diff < 0):
                correct_predictions += 1
                correct_trades += 1

        except (ValueError, TypeError):
            continue

    accuracy_all = (correct_predictions / total_decisions * 100.0) if total_decisions > 0 else 0.0
    accuracy_traded = (correct_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return accuracy_all, accuracy_traded, total_decisions, total_trades, correct_trades


def calculate_pnl_metrics(decisions: List[Tuple]):
    """Calculate PnL-related metrics"""
    total_pnl = 0.0
    total_pnl_traded = 0.0
    pnl_values = []
    pnl_traded_values = []

    for decision in decisions:
        _, _, _, _, _, _, _, pnl_change, _ = decision

        try:
            pnl = float(pnl_change) if pnl_change is not None else 0.0
            total_pnl += pnl
            pnl_values.append(pnl)

            # Only count non-zero PnL changes as trades
            if pnl != 0.0:
                total_pnl_traded += pnl
                pnl_traded_values.append(pnl)
        except (ValueError, TypeError):
            continue

    avg_pnl_per_decision = total_pnl / len(decisions) if decisions else 0.0
    avg_pnl_per_trade = total_pnl_traded / len(pnl_traded_values) if pnl_traded_values else 0.0

    return total_pnl, total_pnl_traded, avg_pnl_per_decision, avg_pnl_per_trade


def calculate_risk_metrics(pnl_values: List[float]):
    """Calculate risk-adjusted return metrics"""
    if not pnl_values:
        return 0.0, 0.0, 0.0

    import numpy as np
    pnl_array = np.array(pnl_values)

    # Volatility (standard deviation)
    volatility = np.std(pnl_array)

    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = np.mean(pnl_array) / volatility if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = np.cumsum(pnl_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    return max_drawdown, sharpe_ratio, volatility


def analyze_action_distribution(decisions: List[Tuple]):
    """Analyze the distribution of actions taken by the agent"""
    action_counts = {}

    for decision in decisions:
        _, _, _, action_str, _, _, _, _, _ = decision
        try:
            action = int(action_str)
            action_counts[action] = action_counts.get(action, 0) + 1
        except (ValueError, TypeError):
            continue

    return action_counts


def calculate_directional_accuracy_including_holds(decisions: List[Tuple], horizon_hours: int, conn: sqlite3.Connection, hold_threshold: float = 0.0005):
    """Directional accuracy counting holds as correct when |future return| < hold_threshold."""
    cursor = conn.cursor()
    total = len(decisions)
    correct = 0
    total_trades = 0

    action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}

    for decision in decisions:
        _, symbol, dec_ts_str, action_str, _, _, _, _, _ = decision
        try:
            dec_ts = parse_iso_utc(dec_ts_str)
            action = int(action_str)
            executed_delta = float(action_map.get(action, 0))
        except (ValueError, TypeError):
            continue

        start_bar = fetch_first_bar_at_or_after(cursor, symbol, dec_ts)
        if not start_bar:
            continue
        start_ts_str, price_start = start_bar
        start_ts = parse_iso_utc(start_ts_str)
        end_target = start_ts + timedelta(hours=horizon_hours)
        end_bar = fetch_first_bar_at_or_after(cursor, symbol, end_target)
        if not end_bar:
            continue
        _, price_end = end_bar
        price_start_f = float(price_start)
        price_end_f = float(price_end)
        ret = (price_end_f - price_start_f) / price_start_f

        if executed_delta == 0:
            # hold is correct if small absolute return
            if abs(ret) < hold_threshold:
                correct += 1
        else:
            total_trades += 1
            if (executed_delta > 0 and ret > 0) or (executed_delta < 0 and ret < 0):
                correct += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    return acc


def evaluate_agent_performance(symbol: Optional[str] = None, horizon_hours: int = 1,
                              start: Optional[datetime] = None, end: Optional[datetime] = None):
    """Main evaluation function for SL agent performance"""

    conn, cursor = TradeDatabase.sql_connect()
    if conn is None:
        raise RuntimeError("Could not connect to SQLite database.")

    try:
        # Fetch decisions
        decisions = fetch_decisions(conn, symbol, start, end)

        if not decisions:
            print("No SL decisions found for the given filters.")
            return

        print(f"Evaluating {len(decisions)} SL decisions...")

        # Directional accuracy
        acc_all, acc_traded, total_dec, total_trades, correct_trades = calculate_directional_accuracy(
            decisions, horizon_hours, conn
        )

        # PnL metrics
        total_pnl, total_pnl_traded, avg_pnl_dec, avg_pnl_trade = calculate_pnl_metrics(decisions)

        # Risk metrics
        pnl_values = [float(d[7]) if d[7] is not None else 0.0 for d in decisions]
        max_dd, sharpe, vol = calculate_risk_metrics(pnl_values)

        # Action distribution
        action_dist = analyze_action_distribution(decisions)

        hold_threshold = float(os.environ.get("SL_EVAL_HOLD_THRESHOLD", "0.0005"))
        acc_incl_holds = calculate_directional_accuracy_including_holds(decisions, horizon_hours, conn, hold_threshold)

        # Print results
        print("\n" + "="*60)
        print("SL AGENT EVALUATION SUMMARY")
        print("="*60)

        if symbol:
            print(f"Symbol: {symbol}")
        print(f"Evaluation Horizon: {horizon_hours} hour(s)")
        print(f"Total Decisions: {total_dec}")
        print(f"Total Trades: {total_trades}")
        print(f"Total Holds: {total_dec - total_trades}")

        print("\n" + "-"*40)
        print("ACCURACY METRICS")
        print("-"*40)
        print(f"Directional Accuracy (All): {acc_all:.2f}%")
        print(f"Directional Accuracy (Traded): {acc_traded:.2f}%")
        print(f"Directional Accuracy (Incl. holds<±{hold_threshold:.4f}): {acc_incl_holds:.2f}%")

        print("\n" + "-"*40)
        print("PERFORMANCE METRICS")
        print("-"*40)
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total PnL (Traded): ${total_pnl_traded:.2f}")
        print(f"Average PnL per Decision: ${avg_pnl_dec:.2f}")
        print(f"Average PnL per Trade: ${avg_pnl_trade:.2f}")

        print("\n" + "-"*40)
        print("RISK METRICS")
        print("-"*40)
        print(f"Maximum Drawdown: ${max_dd:.2f}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Volatility: ${vol:.2f}")

        print("\n" + "-"*40)
        print("ACTION DISTRIBUTION")
        print("-"*40)
        action_names = {0: "Strong Sell", 1: "Sell", 2: "Hold", 3: "Buy", 4: "Strong Buy"}
        for action, count in action_dist.items():
            action_name = action_names.get(action, f"Action {action}")
            percentage = (count / total_dec * 100) if total_dec > 0 else 0
            print(f"{action_name}: {count} ({percentage:.1f}%)")

    finally:
        TradeDatabase.close_connection(conn)


def main():
    """Main entry point for the SL evaluation script"""
    parser = argparse.ArgumentParser(
        description="SL Agent Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation for a symbol
  python supervised/sl_evaluate_agent.py --symbol "/ES:XCME{=h}"
  
  # Evaluation with time range
  python supervised/sl_evaluate_agent.py --symbol "/ES:XCME{=h}" --start "2024-01-01" --end "2024-01-31"
  
  # Custom evaluation horizon
  python supervised/sl_evaluate_agent.py --symbol "/ES:XCME{=h}" --horizon 4
        """
    )

    parser.add_argument("--symbol", type=str, help="Symbol to evaluate (e.g., '/ES:XCME{=h}')")
    parser.add_argument("--horizon", type=int, default=1, help="Evaluation horizon in hours (default: 1)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD or ISO format)")

    args = parser.parse_args()

    # Parse dates
    start_dt = None
    end_dt = None

    if args.start:
        try:
            if len(args.start) == 10:  # YYYY-MM-DD format
                start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                start_dt = parse_iso_utc(args.start)
        except ValueError:
            print(f"Invalid start date format: {args.start}")
            return

    if args.end:
        try:
            if len(args.end) == 10:  # YYYY-MM-DD format
                end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                end_dt = parse_iso_utc(args.end)
        except ValueError:
            print(f"Invalid end date format: {args.end}")
            return

    # Run evaluation
    evaluate_agent_performance(
        symbol=args.symbol,
        horizon_hours=args.horizon,
        start=start_dt,
        end=end_dt
    )


if __name__ == "__main__":
    main()
