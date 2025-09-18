import os
import sys
import json
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supervised.infer_supervised import load_model, predict_action
from rl.env import TradingEnv, EnvConfig  # we still reuse env for state building
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


def load_latest_model(symbol: str):
    """
    Load the latest trained SL model (automatically finds the most recent one).
    """
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    
    # Look for the latest lightgbm_sl model
    lightgbm_models = [f for f in os.listdir(model_dir) if f.startswith('lightgbm_sl_') and f.endswith('.txt')]
    
    if not lightgbm_models:
        # Fallback to old baseline name
        model_path = os.path.join(model_dir, "lightgbm_baseline.txt")
        if os.path.exists(model_path):
            model, meta, feature_cols = load_model(model_path, model_type="lightgbm")
            return model, feature_cols, meta, os.path.basename(model_path).split(".")[0]
        else:
            raise FileNotFoundError(f"No supervised model found in {model_dir}")
    
    # Sort by name (which includes timestamp) and get the latest
    lightgbm_models.sort()
    latest_model = lightgbm_models[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    model, meta, feature_cols = load_model(model_path, model_type="lightgbm")
    return model, feature_cols, meta, os.path.basename(model_path).split(".")[0]


def normalize_rl_symbol(rl_symbol: str) -> str:
    if "/" in rl_symbol and ":" in rl_symbol:
        return rl_symbol.split(":")[0]
    elif "/" in rl_symbol and "USD" in rl_symbol:
        return rl_symbol.split("/")[0]
    else:
        return rl_symbol


def get_snapshot_position(rl_symbol: str) -> float:
    conn, cursor = TradeDatabase.sql_connect()
    lookup = normalize_rl_symbol(rl_symbol)
    sql_param = lookup + "%"
    query = """
        SELECT quantity
        FROM positions
        WHERE underlying_symbol LIKE ?
    """
    cursor.execute(query, (sql_param,))
    result = cursor.fetchone()
    TradeDatabase.close_connection(conn)
    return float(result[0]) if result else 0.0


def log_decision(
    decision_id: str,
    symbol: str,
    action: int,
    model_version: str,
    state: np.ndarray,
    info: dict,
    position_before: float,
    confidence: float = 1.0,
    comment: str = "sl-auto",
) -> str:
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute(
        """
        INSERT INTO sl_decisions (
            id, symbol, decision_timestamp, state_json, action, position_before, position_after,
            price, pnl_change, model_version, confidence, comment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            symbol,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(state.tolist()),
            str(action),
            float(position_before),
            float(info.get("position", 0.0)),
            float(info.get("price_prev", 0.0)),
            float(info.get("cash_pnl", 0.0)),
            model_version,
            float(confidence),
            comment,
        ),
    )
    conn.commit()
    TradeDatabase.close_connection(conn)
    return decision_id


def run_once(symbol: str):
    model, feature_cols, meta, model_version = load_latest_model(symbol)
    db_symbol = canonicalize_symbol_for_db(symbol)
    initial_position = get_snapshot_position(symbol)
    print(f"[SL] Current snapshot position for {symbol}: {initial_position}")

    env = TradingEnv(EnvConfig(symbol=db_symbol, include_sentiment=True, initial_position=initial_position))
    obs, _ = env.reset()

    # Align env features to model feature columns
    env_n_features = len(getattr(env, 'feature_cols', []))
    lookback = env.config.lookback_bars
    flat = obs.flatten()
    expected_env_len = env_n_features * lookback + 2  # +position,+cash_pnl in env

    if len(flat) < env_n_features * lookback:
        print("[SL] Warning: observation shorter than expected; falling back to Hold")
        action = 2
        confidence = 0.0
    else:
        # Extract only the feature window part from env obs
        window_part = flat[: env_n_features * lookback]
        window_matrix = window_part.reshape(lookback, env_n_features)
        last_row = window_matrix[-1]
        env_row_dict = dict(zip(env.feature_cols, last_row))

        # Build model input dict in the model's feature order; fill missing with 0.0
        obs_dict = {col: float(env_row_dict.get(col, 0.0)) for col in feature_cols}

        result = predict_action(model, obs_dict, model_type="lightgbm", feature_cols=feature_cols, meta=meta)

        # probability gating (encourage non-hold only when confident)
        prob = float(result.get("prob", 0.0))
        cls = int(result.get("class", 0))  # [-1,0,1]
        min_trade_prob = float(os.environ.get("SL_MIN_TRADE_PROB", "0.6"))

        if cls == 0 or prob < min_trade_prob:
            action = 2  # Hold
        elif cls > 0:
            action = 4  # Buy/Increase
        else:
            action = 0  # Sell/Decrease
        confidence = prob

    next_obs, reward, done, _, info = env.step(int(action))

    decision_id = str(uuid4())
    _ = log_decision(
        decision_id,
        symbol,
        int(action),
        model_version,
        obs,
        info,
        initial_position,
        confidence,
    )

    print(f"[SL] Decision: symbol={symbol} action={int(action)} prob={confidence:.2f}")


if __name__ == "__main__":
    symbol_env = os.environ.get("SL_SYMBOL")
    if symbol_env:
        run_once(symbol_env)
    else:
        for sym in SYMBOLS:
            run_once(sym)
