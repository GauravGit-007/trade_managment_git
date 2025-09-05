import os
import sys
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase
from models.lstm import process_candles_with_ta


@dataclass
class EnvConfig:
    db_name: str = "trade_data.db"
    symbol: str = "/ES:XCME{=h}"
    lookback_bars: int = 8
    include_ta: bool = True
    include_lstm_pred: bool = True
    include_sentiment: bool = True
    initial_position: float = 0.0
    risk_aversion: float = 0.0
    max_position: float = 3.0
    transaction_cost_bps: float = 1.0


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.conn, self.cursor = TradeDatabase.sql_connect(config.db_name)
        if self.conn is None:
            raise RuntimeError("Could not connect to SQLite database for env")

        # Load data for one symbol
        self.df = self._load_joined_dataframe(config.symbol)
        if len(self.df) < config.lookback_bars + 2:
            raise ValueError("Not enough data to run the environment")

        # Build observation vector definition
        obs_dim = 0
        self.obs_slices: Dict[str, slice] = {}

        base_cols = ["open", "high", "low", "close", "volume",
                     "rsi_14", "ema_21", "ema_50",
                     "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
                     "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
                     "atr_14", "STOCHk_14_3_3", "STOCHd_14_3_3"] if self.config.include_ta else []
        pred_cols = ["predicted_value"] if self.config.include_lstm_pred else []
        sent_cols = ["sentiment_score_pos", "sentiment_score_neg", "sentiment_score_neu"] if self.config.include_sentiment else []

        self.feature_cols = base_cols + pred_cols + sent_cols
        obs_dim = len(self.feature_cols) * self.config.lookback_bars + 2  # + position, + cash_pnl

        high = np.full((obs_dim,), np.inf, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: {-1: sell/decrease, 0: hold, +1: buy/increase}
        self.action_space = spaces.Discrete(5)  # -2, -1, 0, +1, +2 step units

        self.reset(seed=None)

    

    def _load_joined_dataframe(self, symbol: str):
        q = """
        WITH base AS (
            SELECT symbol,
                   open, high, low, close, volume,
                   strftime('%Y-%m-%d %H:00:00', timestamp) AS ts
            FROM historical_data_1h
            WHERE symbol = ?
            ORDER BY ts
        ), preds AS (
            SELECT symbol,
                   strftime('%Y-%m-%d %H:00:00', target_timestamp) AS ts,
                   predicted_value
            FROM lstm_predictions
            WHERE symbol = ?
        )
        SELECT b.symbol,
               b.ts AS timestamp,
               b.open, b.high, b.low, b.close, b.volume,
               p.predicted_value
        FROM base b
        LEFT JOIN preds p ON p.symbol = b.symbol AND p.ts = b.ts
        ORDER BY b.ts
        """
        self.cursor.execute(q, (symbol, symbol))
        rows = self.cursor.fetchall()
   
        cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "predicted_value"]
        df = pd.DataFrame(rows, columns=cols)

        # Forward/backfill predictions if missing
        df["predicted_value"] = df["predicted_value"].ffill().bfill()

        # ✅ Reuse LSTM’s TA pipeline here
        df = process_candles_with_ta(df)

        # Optionally join DAILY sentiment aggregates per instrument (match per-day across intraday bars)
        if self.config.include_sentiment:
            try:
                instrument = self._map_symbol_to_instrument(symbol)
                sentiment_sql = """
                SELECT 
                    date(na.published_at) AS day,
                    SUM(CASE WHEN LOWER(sa.sentiment_label) = 'positive' THEN sa.sentiment_score ELSE 0 END) AS sentiment_score_pos,
                    SUM(CASE WHEN LOWER(sa.sentiment_label) = 'negative' THEN sa.sentiment_score ELSE 0 END) AS sentiment_score_neg,
                    SUM(CASE WHEN LOWER(sa.sentiment_label) = 'neutral' THEN sa.sentiment_score ELSE 0 END) AS sentiment_score_neu
                FROM news_articles na
                JOIN sentiment_analysis sa ON sa.article_id = na.id
                WHERE na.instrument LIKE ?
                GROUP BY day
                ORDER BY day
                """
                like_arg = f"%{instrument}%"
                self.cursor.execute(sentiment_sql, (like_arg,))
                sent_rows = self.cursor.fetchall()
                if sent_rows:
                    sent_df = pd.DataFrame(sent_rows, columns=["day", "sentiment_score_pos", "sentiment_score_neg", "sentiment_score_neu"])
                    df["day"] = df["timestamp"].str[:10]
                    df = df.merge(sent_df, on="day", how="left")
                    df.drop(columns=["day"], inplace=True)
                else:
                    df["sentiment_score_pos"] = 0.0
                    df["sentiment_score_neg"] = 0.0
                    df["sentiment_score_neu"] = 0.0
            except Exception as e:
                # Fail-safe: if sentiment join fails, create zero columns
                df["sentiment_score_pos"] = 0.0
                df["sentiment_score_neg"] = 0.0
                df["sentiment_score_neu"] = 0.0

        # ✅ Instead of dropna, fill missing values
        df = df.bfill().ffill()

        # Debug check
        if len(df) < self.config.lookback_bars + 2:
            print("DEBUG: Loaded rows:", len(df))
            print("DEBUG: Example timestamps:", df["timestamp"].head(10).tolist())
            print("DEBUG: Example preds:", df["predicted_value"].head(10).tolist())

        return df



    def _get_observation(self, idx: int) -> np.ndarray:
        start = idx - self.config.lookback_bars + 1
        end = idx + 1
        window = self.df.iloc[start:end]
        feature_matrix = window[self.feature_cols].values.astype(np.float32)
        flat = feature_matrix.flatten()
        obs = np.concatenate([flat, np.array([self.position, self.cash_pnl], dtype=np.float32)])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.position = float(self.config.initial_position)
        self.cash_pnl = 0.0
        self.current_index = self.config.lookback_bars - 1
        obs = self._get_observation(self.current_index)
        return obs, {}

    def step(self, action: int):
        action_map = {
            0: -2.0,
            1: -1.0,
            2: 0.0,
            3: +1.0,
            4: +2.0,
        }
        delta_units = action_map.get(int(action), 0.0)
        prev_price = float(self.df.loc[self.current_index, "close"]) 
        # Apply transaction costs
        transaction_cost = abs(delta_units) * prev_price * (self.config.transaction_cost_bps / 10000.0)

        # Position bounds
        new_position = np.clip(self.position + delta_units, -self.config.max_position, self.config.max_position)
        executed_delta = new_position - self.position
        self.position = float(new_position)

        # Move time
        self.current_index += 1
        done = self.current_index >= len(self.df) - 1
        next_price = float(self.df.loc[self.current_index, "close"]) 

        # Mark-to-market PnL on existing position
        price_change = next_price - prev_price
        mtm = self.position * price_change
        self.cash_pnl += mtm - transaction_cost

        # Simple risk penalty via variance proxy
        risk_penalty = self.config.risk_aversion * (self.position ** 2)
        reward = mtm - transaction_cost - risk_penalty

        obs = self._get_observation(self.current_index)

        info = {
            "price_prev": prev_price,
            "price_next": next_price,
            "executed_delta": executed_delta,
            "transaction_cost": transaction_cost,
            "risk_penalty": risk_penalty,
            "position": self.position,
            "cash_pnl": self.cash_pnl,
        }

        return obs, float(reward), bool(done), False, info

    def render(self):
        print(f"t={self.current_index} pos={self.position} cash_pnl={self.cash_pnl:.2f}")

    def close(self):
        try:
            TradeDatabase.close_connection(self.conn)
        except Exception:
            pass

    def _map_symbol_to_instrument(self, symbol: str) -> str:
        s = symbol.upper()
        if "ETH" in s:
            return "Ethereum"
        if "BTC" in s:
            return "Bitcoin"
        if "/ES" in s:
            return "S&P 500"
        if "/NQ" in s:
            return "NASDAQ"
        if "/RTY" in s:
            return "Russell 2000"
        if "/QM" in s:
            return "Crude Oil"
        if "/QG" in s:
            return "Natural Gas"
        return "GENERAL"

