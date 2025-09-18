"""
Time-series Dataset for supervised signal prediction.
Assumes processed candles CSV with a datetime index and columns including:
['timestamp','open','high','low','close','volume', ... technical indicators ...]
Labels should be produced by scripts/make_labels.py and saved as a column 'label_{h}' where h is minutes horizon.
""" 
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, candles_csv: str, label_col: str, feature_cols: Optional[List[str]] = None,
                 window: int = 64, stride: int = 1, transform=None):
        self.df = pd.read_csv(candles_csv, parse_dates=['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        if feature_cols is None:
            # use all numeric columns except label columns
            exclude = [c for c in self.df.columns if c.startswith('label_') or c in ('timestamp',)]
            feature_cols = [c for c in self.df.columns if c not in exclude and np.issubdtype(self.df[c].dtype, np.number)]
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window = window
        self.stride = stride
        self.transform = transform

        # precompute indices for valid windows (no NaNs in window or label)
        self.valid_indices = []
        n = len(self.df)
        for end in range(window-1, n, stride):
            start = end - (window-1)
            label_idx = end  # label associated with the last row of the window
            if pd.isna(self.df.loc[label_idx, label_col]):
                continue
            window_df = self.df.iloc[start:end+1]
            if window_df[self.feature_cols].isnull().values.any():
                continue
            self.valid_indices.append((start, end))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start, end = self.valid_indices[idx]
        window_df = self.df.iloc[start:end+1]
        X = window_df[self.feature_cols].values.astype('float32')  # shape (window, n_features)
        y = float(self.df.loc[end, self.label_col])
        # return as tensors: (C, L) typical for conv1d/torch: transpose to (features, window)
        X_t = torch.from_numpy(X).permute(1,0).contiguous()   # (n_features, window)
        y_t = torch.tensor(y, dtype=torch.float32)
        if self.transform:
            X_t = self.transform(X_t)
        return X_t, y_t

if __name__ == "__main__":
    # quick sanity check when run directly
    ds = TimeSeriesDataset(candles_csv='../db/processed_candles_with_ta.csv', label_col='label_15', window=32)
    print('dataset len', len(ds))
    x,y = ds[0]
    print('sample shapes', x.shape, y.shape)