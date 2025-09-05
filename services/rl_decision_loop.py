import os
import sys
import json
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import TradingEnv, EnvConfig
from db.database import TradeDatabase


SYMBOLS = [
    "/NQ:XCME",
    "/ES:XCME",
    "/RTY:XCME",
    "/QG:XNYM",
    "/QM:XNYM",
    "BTC/USD:CXTALP",
    "ETH/USD:CXTALP",
    "/MES:XCME",
    "/MNQ:XCME",
    "/MCL:XNYM",
]


def canonicalize_symbol_for_db(symbol: str) -> str:
    return symbol if "{=" in symbol else f"{symbol}{{=h}}"


def sanitize_symbol(symbol: str) -> str:
    return (
        symbol.replace("/", "_")
        .replace(":", "-")
        .replace("{", "")
        .replace("}", "")
        .replace("=", "")
    )


def load_latest_model(symbol: str) -> tuple[PPO, str]:
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    sym_tag = sanitize_symbol(canonicalize_symbol_for_db(symbol))
    models = [f for f in os.listdir(outputs_dir) if f.endswith('.zip') and f"ppo_{sym_tag}_" in f]
    if not models:
        # fallback: load any latest if per-symbol not found
        models = [f for f in os.listdir(outputs_dir) if f.endswith('.zip')]
        if not models:
            raise FileNotFoundError("No trained PPO model found in outputs/")
    models.sort()
    latest = models[-1]
    model_path = os.path.join(outputs_dir, latest)
    model = PPO.load(model_path)
    return model, os.path.splitext(latest)[0]

def normalize_rl_symbol(rl_symbol: str) -> str:
    """
    Convert RL symbols like '/MES:XCME' into a prefix for positions.underlying_symbol
    Example:
        /MES:XCME -> /MES
        /MNQ:XCME -> /MNQ
        /MCL:XNYM -> /MCL
        BTC/USD:CXTALP -> BTC  (crypto case)
    """
    if "/" in rl_symbol and ":" in rl_symbol:
        # futures format /MES:XCME -> /MES
        return rl_symbol.split(":")[0]
    elif "/" in rl_symbol and "USD" in rl_symbol:
        # crypto case BTC/USD:CXTALP -> BTC
        return rl_symbol.split("/")[0]
    else:
        return rl_symbol

### CHANGED: new helper to snapshot position directly from `positions`
def get_snapshot_position(rl_symbol: str) -> float:
    conn, cursor = TradeDatabase.sql_connect()
    lookup = normalize_rl_symbol(rl_symbol)   # already returns with %
    sql_param = lookup + "%"
    #print(f"[DEBUG] rl_symbol={rl_symbol!r}, normalized={lookup!r}, sql_param={sql_param!r}")

    query = """
        SELECT quantity
        FROM positions
        WHERE underlying_symbol LIKE ?
    """
    cursor.execute(query, (sql_param,))
    result = cursor.fetchone()

    TradeDatabase.close_connection(conn)

    qty = result[0] if result else 0.0
    #print(f"[DEBUG] Query result for {lookup}: {qty}")
    return float(qty)



def log_decision(
    decision_id: str,
    symbol: str,
    action: int,
    policy_version: str,
    state: np.ndarray,
    info: dict,
    position_before: float,
    confidence: float = 1.0,
    comment: str = "auto",
) -> str:
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute(
        """
        INSERT INTO rl_decisions (
            id, symbol, decision_timestamp, state_json, action, position_before, position_after,
            price, pnl_change, policy_version, confidence, comment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            symbol,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(state.tolist()),
            str(action),
            float(position_before),                # <-- snapshot
            float(info.get("position", 0.0)),      # <-- env position_after
            float(info.get("price_prev", 0.0)),
            float(info.get("cash_pnl", 0.0)),
            policy_version,
            float(confidence),
            comment,
        ),
    )
    conn.commit()
    TradeDatabase.close_connection(conn)
    return decision_id


def run_once(symbol: str):
    model, policy_version = load_latest_model(symbol)

    db_symbol = canonicalize_symbol_for_db(symbol)

    ### CHANGED: snapshot live position from positions table
    initial_position = get_snapshot_position(symbol)
    print(f"[DEBUG] Current snapshot position for {symbol}: {initial_position}")

    env = TradingEnv(EnvConfig(symbol=db_symbol, include_sentiment=True, initial_position=initial_position))
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, _, info = env.step(int(action))

    decision_id = str(uuid4())
    _ = log_decision(
        decision_id,
        symbol,
        int(action),
        policy_version,
        obs,
        info,
        initial_position,   # <-- snapshot passed as before
    )

    # Experience logging
    try:
        conn, cursor = TradeDatabase.sql_connect()
        cursor.execute(
            """
            INSERT INTO rl_experiences (
                id, symbol, t_timestamp, state_json, action, reward, next_state_json, done, episode_id, info_json,
                position_before, position_after, executed_delta, price_prev, price_next, transaction_cost, risk_penalty, cash_pnl, policy_version, decision_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                symbol,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(obs.tolist()),
                int(action),
                float(reward),
                json.dumps(next_obs.tolist()),
                int(done),
                None,
                json.dumps(info),
                float(initial_position),               # <-- snapshot
                float(info.get("position", 0.0)),      # <-- env position_after
                float(info.get("executed_delta", 0.0)),
                float(info.get("price_prev", 0.0)),
                float(info.get("price_next", 0.0)),
                float(info.get("transaction_cost", 0.0)),
                float(info.get("risk_penalty", 0.0)),
                float(info.get("cash_pnl", 0.0)),
                policy_version,
                decision_id,
            ),
        )
        conn.commit()
        TradeDatabase.close_connection(conn)
    except Exception:
        pass
    print(f"Decision: symbol={symbol} action={int(action)} reward={float(reward):.4f}")


if __name__ == "__main__":
    symbol_env = os.environ.get("RL_SYMBOL")
    if symbol_env:
        run_once(symbol_env)
    else:
        for sym in SYMBOLS:
            run_once(sym)
