"""
Train a simple PyTorch temporal CNN/LSTM model.
Saves model to models/supervised_signal.pt
Usage:
    python supervised/train_pytorch.py --candles db/processed_candles_with_ta.csv --labels data/processed_with_labels_15m.parquet --label_col label_15 --model_out models/supervised_signal.pt
"""
import argparse, os, json
import pandas as pd, numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from supervised.dataset import TimeSeriesDataset

class SmallCNN(nn.Module):
    def __init__(self, n_features, window, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        # x: (batch, features, window)
        h = torch.relu(self.conv1(x))
        h = self.pool(h).squeeze(-1)
        out = self.fc(h)
        return out

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for X,y in loader:
        X = X.to(device); y = y.to(device).long() + 1  # convert -1,0,1 to 0..2
        opt.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X,y in loader:
            out = model(X.to(device))
            p = out.argmax(dim=1).cpu().numpy()
            preds.append(p)
            trues.append((y.numpy()+1).astype(int))
    import numpy as np
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    from sklearn.metrics import classification_report, accuracy_score
    print('Acc', accuracy_score(trues, preds))
    print(classification_report(trues, preds))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candles', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--label_col', required=True)
    p.add_argument('--model_out', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--window', type=int, default=32)
    args = p.parse_args()

    ds = TimeSeriesDataset(args.candles, args.label_col, window=args.window)
    # split by time: first 80% train, last 20% test
    n = len(ds); n_train = int(0.8 * n)
    train_ds = torch.utils.data.Subset(ds, range(0, n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    sample_x, _ = ds[0]
    n_features = sample_x.shape[0]
    window = sample_x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCNN(n_features, window, n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, criterion, device)
        print(f'Epoch {epoch} loss {loss:.4f}')
        evaluate(model, val_loader, device)
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'n_features': n_features, 'window': window}, args.model_out)
    print('Saved', args.model_out)

if __name__ == '__main__':
    main()