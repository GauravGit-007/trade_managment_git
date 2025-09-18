"""
Train a baseline LightGBM classifier on engineered features.
Saves model to models/lightgbm_baseline.txt and a small wrapper for inference.
Usage:
    python supervised/train_baseline.py --candles db/processed_candles_with_ta.csv --labels data/processed_with_labels_15m.parquet --label_col label_15 --model_out models/lightgbm_baseline.txt
"""
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

def prepare_features(df, label_col, feature_cols=None):
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if feature_cols is None:
        exclude = [c for c in df.columns if c.startswith('label_') or c in ('timestamp',)]
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y, feature_cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candles', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--label_col', required=True)
    p.add_argument('--model_out', required=True)
    p.add_argument('--test_size', type=float, default=0.2)
    args = p.parse_args()

    df_c = pd.read_csv(args.candles, parse_dates=['timestamp'])
    df_l = pd.read_parquet(args.labels)
    df = df_c.merge(df_l[['timestamp', args.label_col]], on='timestamp', how='left')
    X, y, feature_cols = prepare_features(df, args.label_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=200, early_stopping_rounds=20)
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    model.save_model(args.model_out)
    # save metadata (feature_cols)
    joblib.dump({'feature_cols': feature_cols}, args.model_out + '.meta.pkl')
    preds = np.argmax(model.predict(X_test), axis=1) - 1  # LightGBM returns 0..2; map to -1,0,1
    print('Accuracy', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

if __name__ == '__main__':
    main()