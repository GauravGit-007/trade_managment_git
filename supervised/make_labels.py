"""
Labeling script: compute future returns and convert to classification labels.
Saves label columns to data/labels_{h}.parquet and also appends to original CSV (recommended to write a new file).
Usage:
    python supervised/make_labels.py --candles db/processed_candles_with_ta.csv --horizon 15 --out data/processed_with_labels_15m.parquet --threshold 0.001 --cost 0.0005
"""
import argparse
import pandas as pd
import numpy as np
import os

def make_labels(df: pd.DataFrame, horizon: int, price_col: str='close', threshold: float=0.001, cost: float=0.0005):
    df = df.sort_values('timestamp').reset_index(drop=True)
    # compute forward close at t+h (h rows ahead - assumes candles are uniform in minutes)
    df[f'future_close_{horizon}'] = df[price_col].shift(-horizon)
    df[f'future_return_{horizon}'] = (df[f'future_close_{horizon}'] - df[price_col]) / df[price_col]
    # subtract round-trip cost when classifying (conservative)
    df[f'future_return_net_{horizon}'] = df[f'future_return_{horizon}'] - cost
    # classify: 1 buy, 0 hold, -1 sell (map to RL action mapping later)
    conditions = [
        (df[f'future_return_net_{horizon}'] > threshold),
        (df[f'future_return_net_{horizon}'] < -threshold)
    ]
    choices = [1, -1]
    df[f'label_{horizon}'] = np.select(conditions, choices, default=0)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candles', required=True)
    p.add_argument('--horizon', type=int, default=15, help='horizon in number of candle steps (not minutes unless candles are minute bars)')
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=float, default=0.001, help='return threshold (e.g., 0.001 = 0.1%)')
    p.add_argument('--cost', type=float, default=0.0005, help='transaction cost per round trip')
    args = p.parse_args()

    df = pd.read_csv(args.candles, parse_dates=['timestamp'])
    df2 = make_labels(df, args.horizon, threshold=args.threshold, cost=args.cost)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df2.to_parquet(args.out, index=False)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()