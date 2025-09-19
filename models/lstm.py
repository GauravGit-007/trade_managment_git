import os
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# --- Local DB helper ---
import sys
# Import toolsDB from your project (adjust import if your path is different)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase 



# =========================
# Config
# =========================
DB_NAME = "trade_data.db"
SEQUENCE_LENGTH = 24          # past 24 hours
FUTURE_STEPS = 12             # predict next 12 hours
TARGET_COL = "close"
FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "ema_21", "ema_50",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
    "atr_14", "STOCHk_14_3_3", "STOCHd_14_3_3"
]
EPOCHS = 100
BATCH_SIZE = 32
MODEL_VERSION = "lstm_v1_multi_symbol_1h"
PLOT = True  # set False on servers without display


# =========================
# 1) Load data from SQLite
# =========================
def load_historical_1h(db_name: str = DB_NAME) -> pd.DataFrame:
    conn, cursor = TradeDatabase.sql_connect(db_name)
    if conn is None:
        raise RuntimeError("Could not connect to SQLite database.")

    query = """
        SELECT symbol, open, high, low, close, volume, timestamp
        FROM historical_data_1h
        ORDER BY symbol, timestamp
    """
    df = pd.read_sql_query(query, conn)
    TradeDatabase.close_connection(conn)

    # normalize/clean
    if df.empty:
        raise ValueError("historical_data_1h returned no rows.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df
    
'''
def process_candles_with_ta(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"])

    all_features = []

    for symbol in df["symbol"].unique():
        print(f"[DEBUG] Processing TA for symbol: {symbol}")  # <--- debug
        sub_df = df[df["symbol"] == symbol].copy()
        sub_df = sub_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        sub_df.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")

        # Drop rows where any OHLC is missing
        sub_df = sub_df.dropna(subset=["open", "high", "low", "close"])
        # Fill missing volume with 0
        sub_df["volume"] = sub_df["volume"].fillna(0)

        sub_df["rsi_14"] = ta.rsi(sub_df["close"], length=14)
        sub_df["ema_21"] = ta.ema(sub_df["close"], length=21)
        sub_df["ema_50"] = ta.ema(sub_df["close"], length=50)

        macd = ta.macd(sub_df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            sub_df = sub_df.join(macd)

        bbands = ta.bbands(sub_df["close"], length=20, std=2)
        if bbands is not None:
            sub_df = sub_df.join(bbands)

        sub_df["atr_14"] = ta.atr(high=sub_df["high"], low=sub_df["low"], close=sub_df["close"], length=14)
        stoch = ta.stoch(high=sub_df["high"], low=sub_df["low"], close=sub_df["close"], k=14, d=3)
        if stoch is not None:
            sub_df = sub_df.join(stoch)

        print(f"[DEBUG] Rows before dropna for {symbol}: {len(sub_df)}")  # <--- debug
        sub_df = sub_df.dropna()
        print(f"[DEBUG] Rows after dropna for {symbol}: {len(sub_df)}")  # <--- debug

        sub_df["symbol"] = symbol
        sub_df.reset_index(inplace=True)
        all_features.append(sub_df)

    final_df = pd.concat(all_features, ignore_index=True)
    final_df = final_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[DEBUG] Total rows after TA processing: {len(final_df)}")
    return final_df


def scale_and_make_sequences(final_df: pd.DataFrame, features: list, target: str, seq_len: int, future_steps: int):
    scalers = {}
    scaled_list = []

    for sym in final_df["symbol"].unique():
        df_sym = final_df[final_df["symbol"] == sym].copy()
        print(f"[DEBUG] Scaling symbol: {sym}, rows: {len(df_sym)}")  # <--- debug
        scaler = MinMaxScaler()
        try:
            df_sym[features] = scaler.fit_transform(df_sym[features])
        except Exception as e:
            print(f"[ERROR] Scaling failed for {sym}: {e}")
            continue
        scalers[sym] = scaler
        scaled_list.append(df_sym)

    scaled_df = pd.concat(scaled_list, ignore_index=True)
    X, y = [], []

    for sym in scaled_df["symbol"].unique():
        df_sym = scaled_df[scaled_df["symbol"] == sym]
        vals = df_sym[features].values
        print(f"[DEBUG] Generating sequences for {sym}, available rows: {len(vals)}")  # <--- debug
        seq_count = 0
        for i in range(seq_len, len(vals) - future_steps + 1):
            X.append(vals[i - seq_len:i])
            y.append([vals[i + j][features.index(target)] for j in range(future_steps)])
            seq_count += 1
        print(f"[DEBUG] Sequences generated for {sym}: {seq_count}")  # <--- debug

    X, y = np.array(X), np.array(y)
    print(f"[DEBUG] Total sequences X: {len(X)}, y: {len(y)}")
    return scaled_df, scalers, X, y


def predict_next_per_symbol(final_df, scaled_df, scalers, model, features, target, sequence_length, future_steps):
    predictions = {}

    for sym in final_df["symbol"].unique():
        print(f"[DEBUG] Predicting next for symbol: {sym}")  # <--- debug
        df_sym_raw = final_df[final_df["symbol"] == sym].sort_values("timestamp")
        df_sym_scaled = scaled_df[scaled_df["symbol"] == sym].sort_values("timestamp")
        print(f"[DEBUG] Rows scaled for {sym}: {len(df_sym_scaled)}")  # <--- debug

        if len(df_sym_scaled) < sequence_length:
            print(f"[WARN] Skipping {sym}: not enough rows for last sequence ({len(df_sym_scaled)} < {sequence_length})")
            continue

        last_seq = df_sym_scaled[features].values[-sequence_length:]
        last_seq = np.expand_dims(last_seq, axis=0)
        pred_scaled = model.predict(last_seq, verbose=0)[0]

        scaler = scalers[sym]
        t_idx = features.index(target)
        close_min = scaler.data_min_[t_idx]
        close_max = scaler.data_max_[t_idx]
        pred_unscaled = pred_scaled * (close_max - close_min) + close_min

        last_ts = df_sym_raw["timestamp"].iloc[-1]
        fut_timestamps = generate_future_timestamps(last_ts, future_steps)
        fut_iso = [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in fut_timestamps]
        predictions[sym] = list(zip(fut_iso, pred_unscaled.tolist()))
        print(f"[DEBUG] Predictions generated for {sym}: {len(pred_unscaled)} steps")

    print(f"[DEBUG] Total symbols predicted: {len(predictions)}")
    return predictions
'''
# =========================
# 2) TA feature engineering
# =========================

def process_candles_with_ta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"])

    all_features = []

    for symbol in df["symbol"].unique():
        #sub_df = df[df["symbol"] == symbol].copy()
        #sub_df.set_index("timestamp", inplace=True)
        sub_df = df[df["symbol"] == symbol].copy()

        # Ensure timestamps are unique per symbol
        sub_df = sub_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        sub_df.set_index("timestamp", inplace=True)

        # Force numeric conversion for OHLCV columns
        for col in ["open", "high", "low", "close", "volume"]:
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")

        # Drop rows where any OHLC is missing
        sub_df = sub_df.dropna(subset=["open", "high", "low", "close"])
        # Fill missing volume with 0
        sub_df["volume"] = sub_df["volume"].fillna(0)

        sub_df["rsi_14"] = ta.rsi(sub_df["close"], length=14)
        sub_df["ema_21"] = ta.ema(sub_df["close"], length=21)
        sub_df["ema_50"] = ta.ema(sub_df["close"], length=50)

        macd = ta.macd(sub_df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            sub_df = sub_df.join(macd)

        bbands = ta.bbands(sub_df["close"], length=20, std=2)
        if bbands is not None:
            sub_df = sub_df.join(bbands)

        sub_df["atr_14"] = ta.atr(
            high=sub_df["high"], low=sub_df["low"], close=sub_df["close"], length=14
        )

        stoch = ta.stoch(
            high=sub_df["high"], low=sub_df["low"], close=sub_df["close"], k=14, d=3
        )
        if stoch is not None:
            sub_df = sub_df.join(stoch)

        sub_df = sub_df.dropna()
        sub_df["symbol"] = symbol
        sub_df.reset_index(inplace=True)
        all_features.append(sub_df)

    final_df = pd.concat(all_features, ignore_index=True)
    final_df = final_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return final_df


# =========================
# 3) Build & Train LSTM
# =========================
def build_lstm(input_timesteps: int, input_features: int, future_steps: int) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_timesteps, input_features),
             dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.001)),
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.001)),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.001)),
        Dense(future_steps)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def scale_and_make_sequences(final_df: pd.DataFrame,
                             features: list,
                             target: str,
                             seq_len: int,
                             future_steps: int):
    # scale per symbol
    scalers = {}
    scaled_list = []

    for sym in final_df["symbol"].unique():
        df_sym = final_df[final_df["symbol"] == sym].copy()
        scaler = MinMaxScaler()
        df_sym[features] = scaler.fit_transform(df_sym[features])
        scalers[sym] = scaler
        scaled_list.append(df_sym)

    scaled_df = pd.concat(scaled_list, ignore_index=True)

    # sequences
    X, y = [], []
    for sym in scaled_df["symbol"].unique():
        df_sym = scaled_df[scaled_df["symbol"] == sym]
        vals = df_sym[features].values
        for i in range(seq_len, len(vals) - future_steps + 1):
            X.append(vals[i - seq_len:i])
            y.append([vals[i + j][features.index(target)] for j in range(future_steps)])
    X, y = np.array(X), np.array(y)
    return scaled_df, scalers, X, y


def train_lstm(final_df: pd.DataFrame,
               features: list = FEATURES,
               target: str = TARGET_COL,
               sequence_length: int = SEQUENCE_LENGTH,
               future_steps: int = FUTURE_STEPS,
               epochs: int = EPOCHS,
               batch_size: int = BATCH_SIZE,
               plot: bool = PLOT):
    final_df = final_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    scaled_df, scalers, X, y = scale_and_make_sequences(
        final_df, features, target, sequence_length, future_steps
    )
    print("X shape:", X.shape, "y shape:", y.shape)
    if len(X) < 10:
        raise ValueError("Not enough data to train. Need more rows per symbol.")

    # split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # model
    model = build_lstm(X.shape[1], X.shape[2], future_steps)

    # train
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # evaluate
    y_pred = model.predict(X_test, verbose=0)
    for i in range(future_steps):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"Step {i+1} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    #if plot:
    #    plt.figure(figsize=(14, 6))
    #    for step in range(future_steps):
    #        plt.plot(y_test[:, step], label=f"True t+{step+1}")
    #        plt.plot(y_pred[:, step], label=f"Pred t+{step+1}", linestyle="--")
    #    plt.legend()
    #    plt.title(f"LSTM Predictions (Next {future_steps} hours)")
    #    plt.show()

    return model, scalers, scaled_df


# =========================
# 4) Predict & store to DB
# =========================
def generate_future_timestamps(last_ts: pd.Timestamp, steps: int, freq: str = "H"):
    # Ensure UTC and naive ISO 'Z'
    base = last_ts.tz_convert("UTC") if last_ts.tzinfo is not None else last_ts.tz_localize("UTC")
    return [ (base + pd.Timedelta(hours=i+1)) for i in range(steps) ]


def predict_next_per_symbol(final_df: pd.DataFrame,
                            scaled_df: pd.DataFrame,
                            scalers: dict,
                            model: Sequential,
                            features: list,
                            target: str,
                            sequence_length: int,
                            future_steps: int):
    predictions = {}  # symbol -> list of (target_ts_iso, predicted_value)

    for sym in final_df["symbol"].unique():
        df_sym_raw = final_df[final_df["symbol"] == sym].sort_values("timestamp")
        df_sym_scaled = scaled_df[scaled_df["symbol"] == sym].sort_values("timestamp")

        if len(df_sym_scaled) < sequence_length:
            print(f"[WARN] Skipping {sym}: not enough rows for last sequence.")
            continue

        last_seq = df_sym_scaled[features].values[-sequence_length:]
        last_seq = np.expand_dims(last_seq, axis=0)
        pred_scaled = model.predict(last_seq, verbose=0)[0]

        # inverse-scale ONLY the target dimension (close)
        scaler = scalers[sym]
        t_idx = features.index(target)
        close_min = scaler.data_min_[t_idx]
        close_max = scaler.data_max_[t_idx]
        pred_unscaled = pred_scaled * (close_max - close_min) + close_min

        # future timestamps (hourly)
        last_ts = df_sym_raw["timestamp"].iloc[-1]
        fut_timestamps = generate_future_timestamps(last_ts, future_steps)

        # ISO8601 with Z
        fut_iso = [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in fut_timestamps]
        predictions[sym] = list(zip(fut_iso, pred_unscaled.tolist()))

    return predictions


def upsert_predictions_to_db(predictions: dict,
                             model_version: str = MODEL_VERSION,
                             db_name: str = DB_NAME,
                             is_historical: int = 0):
    conn, cursor = TradeDatabase.sql_connect(db_name)
    if conn is None:
        raise RuntimeError("Could not connect to SQLite database.")

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lstm_predictions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            target_timestamp TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            actual_value REAL,
            model_version TEXT,
            is_historical INTEGER DEFAULT 0,
            UNIQUE(symbol, target_timestamp)
        );
    """)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # SQLite UPSERT (8 columns now, not 7)
    sql = """
        INSERT INTO lstm_predictions
            (id, symbol, prediction_timestamp, target_timestamp, predicted_value, actual_value, model_version, is_historical)
        VALUES
            (?, ?, ?, ?, ?, NULL, ?, ?)
        ON CONFLICT(symbol, target_timestamp) DO UPDATE SET
            prediction_timestamp=excluded.prediction_timestamp,
            predicted_value=excluded.predicted_value,
            model_version=excluded.model_version,
            is_historical=excluded.is_historical
    """

    rows = []
    for sym, pairs in predictions.items():
        for target_ts, value in pairs:
            rows.append((
                str(uuid4()), sym, now_iso, target_ts, float(value), model_version, is_historical
            ))

    cursor.executemany(sql, rows)
    conn.commit()
    TradeDatabase.close_connection(conn)
    print(f"Upserted {len(rows)} prediction rows into lstm_predictions (is_historical={is_historical}).")



def predict_historical_per_symbol(final_df: pd.DataFrame,
                                  scaled_df: pd.DataFrame,
                                  scalers: dict,
                                  model: Sequential,
                                  features: list,
                                  target: str,
                                  sequence_length: int,
                                  future_steps: int):
    """
    Generate rolling predictions across the full historical dataset,
    simulating how the model would have predicted at each point in time.
    """
    predictions = {}

    for sym in final_df["symbol"].unique():
        df_sym_raw = final_df[final_df["symbol"] == sym].sort_values("timestamp")
        df_sym_scaled = scaled_df[scaled_df["symbol"] == sym].sort_values("timestamp")

        if len(df_sym_scaled) < sequence_length + future_steps:
            print(f"[WARN] Skipping {sym}: not enough rows for historical predictions.")
            continue

        scaler = scalers[sym]
        t_idx = features.index(target)
        close_min, close_max = scaler.data_min_[t_idx], scaler.data_max_[t_idx]

        preds = []
        vals = df_sym_scaled[features].values

        for i in range(sequence_length, len(vals) - future_steps + 1):
            seq = vals[i - sequence_length:i]
            seq = np.expand_dims(seq, axis=0)

            pred_scaled = model.predict(seq, verbose=0)[0]
            pred_unscaled = pred_scaled * (close_max - close_min) + close_min

            fut_timestamps = df_sym_raw["timestamp"].iloc[i:i+future_steps]
            fut_iso = [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in fut_timestamps]

            preds.extend(list(zip(fut_iso, pred_unscaled.tolist())))

        predictions[sym] = preds

    return predictions

# =========================
# Main pipeline
# =========================
def main():
    # Make sure all tables exist (safe to call)
    TradeDatabase.create_tables()

    print("Loading historical 1h data...")
    raw_df = load_historical_1h(DB_NAME)
    print(f"Loaded {len(raw_df)} rows for {raw_df['symbol'].nunique()} symbols.")

    print("Computing TA features...")
    final_df = process_candles_with_ta(raw_df)

    # Sanity: ensure all required columns exist
    missing = [c for c in FEATURES + ["symbol", "timestamp"] if c not in final_df.columns]
    if missing:
        raise KeyError(f"Missing required columns after TA processing: {missing}")

    print("Training LSTM...")
    model, scalers, scaled_df = train_lstm(
        final_df,
        features=FEATURES,
        target=TARGET_COL,
        sequence_length=SEQUENCE_LENGTH,
        future_steps=FUTURE_STEPS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        plot=PLOT
    )

    
    #print("Predicting historical (backfill) horizons per symbol...")
    #historical_predictions = predict_historical_per_symbol(
    #    final_df=final_df,
    #    scaled_df=scaled_df,
    #    scalers=scalers,
    #    model=model,
    #    features=FEATURES,
    #    target=TARGET_COL,
    #    sequence_length=SEQUENCE_LENGTH,
    #    future_steps=FUTURE_STEPS
    #)

    #print("Writing historical predictions to SQLite...")
    #upsert_predictions_to_db(historical_predictions, MODEL_VERSION, DB_NAME, is_historical=1)

    print("Predicting next (live forward) horizons per symbol...")
    live_predictions = predict_next_per_symbol(
        final_df=final_df,
        scaled_df=scaled_df,
        scalers=scalers,
        model=model,
        features=FEATURES,
        target=TARGET_COL,
        sequence_length=SEQUENCE_LENGTH,
        future_steps=FUTURE_STEPS
    )

    print("Writing live forward predictions to SQLite...")
    upsert_predictions_to_db(live_predictions, MODEL_VERSION, DB_NAME, is_historical=0)


    print("Done.")


if __name__ == "__main__":
    main()
