import pytest, os
from supervised.dataset import TimeSeriesDataset
def test_dataset_loads():
    base = os.path.join('..','db','processed_candles_with_ta.csv')
    ds = TimeSeriesDataset(candles_csv=base, label_col='label_15', window=16)
    assert len(ds) > 0
    x,y = ds[0]
    assert x.shape[0] > 0
