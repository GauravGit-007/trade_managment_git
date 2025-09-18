"""
Comprehensive SL training pipeline that mirrors RL training capabilities.
Handles data preparation, feature engineering, model training, and evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase
from supervised.dataset import TimeSeriesDataset
from supervised.train_pytorch import SmallCNN


class SLTrainingPipeline:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file or use defaults. If no path provided, try supervised/config.yaml."""
        if not config_path:
            default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if os.path.exists(default_cfg):
                config_path = default_cfg
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {
                'labels': {'horizon_minutes': 60, 'threshold': 0.0003, 'cost': 0.0001},
                'dataset': {'window': 32, 'stride': 1, 'test_size': 0.2, 'validation_size': 0.1},
                'training': {
                    'baseline': {
                        'model_type': 'lightgbm',
                        'num_boost_round': 200,
                        'early_stopping_rounds': 20,
                        'learning_rate': 0.05,
                        'num_leaves': 31
                    },
                    'pytorch': {'epochs': 50, 'batch_size': 64, 'lr': 0.001, 'patience': 10}
                }
            }
        # ensure required keys exist
        cfg.setdefault('training', {}).setdefault('baseline', {}).setdefault('num_boost_round', 200)
        cfg['training']['baseline'].setdefault('early_stopping_rounds', 20)
        cfg['training']['baseline'].setdefault('learning_rate', 0.05)
        cfg['training']['baseline'].setdefault('num_leaves', 31)
        cfg.setdefault('dataset', {}).setdefault('validation_size', 0.1)
        cfg['dataset'].setdefault('test_size', 0.2)
        # ensure pytorch defaults
        cfg['training'].setdefault('pytorch', {})
        cfg['training']['pytorch'].setdefault('epochs', 50)
        cfg['training']['pytorch'].setdefault('batch_size', 64)
        cfg['training']['pytorch'].setdefault('lr', 0.001)
        cfg['training']['pytorch'].setdefault('patience', 10)
        return cfg
    
    def prepare_training_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """Prepare comprehensive training data from processed CSV. Symbols optional."""
        conn, cursor = TradeDatabase.sql_connect()
        candles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "processed_candles_with_ta.csv")
        if not os.path.exists(candles_path):
            raise FileNotFoundError(f"Processed candles file not found: {candles_path}")
        df = pd.read_csv(candles_path, parse_dates=['timestamp'])
        if symbols:
            # canonicalize provided symbols to include {=h} if missing
            def canon(s: str) -> str:
                return s if "{" in s else f"{s}{{=h}}"
            wanted = set(canon(s) for s in symbols)
            df = df[df['symbol'].isin(list(wanted))]
        # labels and features
        df = self._add_labels(df)
        df = self._add_engineered_features(df)
        TradeDatabase.close_connection(conn)
        return df
    
    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add classification labels based on future returns."""
        horizon = self.config['labels']['horizon_minutes']
        threshold = self.config['labels']['threshold']
        cost = self.config['labels']['cost']
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute future returns
        df[f'future_close_{horizon}'] = df['close'].shift(-horizon)
        df[f'future_return_{horizon}'] = (df[f'future_close_{horizon}'] - df['close']) / df['close']
        df[f'future_return_net_{horizon}'] = df[f'future_return_{horizon}'] - cost
        
        # Create labels: 1=buy, 0=hold, -1=sell
        conditions = [
            (df[f'future_return_net_{horizon}'] > threshold),
            (df[f'future_return_net_{horizon}'] < -threshold)
        ]
        choices = [1, -1]
        df[f'label_{horizon}'] = np.select(conditions, choices, default=0)
        
        return df
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional engineered features for better model performance."""
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility features
        df['volatility_20'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['close'].rolling(20).mean()
        
        # Momentum features
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Technical indicator ratios
        if 'rsi_14' in df.columns:
            df['rsi_normalized'] = (df['rsi_14'] - 50) / 50
        if 'MACD_12_26_9' in df.columns:
            df['macd_signal_ratio'] = df['MACD_12_26_9'] / (df['MACD_12_26_9'].rolling(9).mean() + 1e-8)
        
        return df
    
    def train_lightgbm_model(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """Train LightGBM model with time series cross-validation."""
        horizon = self.config['labels']['horizon_minutes']
        label_col = f'label_{horizon}'
        
        # Prepare features
        exclude_cols = [c for c in df.columns if c.startswith('label_') or c in ['timestamp', 'symbol']]
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # Remove rows with NaN labels
        df_clean = df.dropna(subset=[label_col]).reset_index(drop=True)
        X = df_clean[feature_cols].values
        y = df_clean[label_col].values
        
        # Convert labels from [-1, 0, 1] to [0, 1, 2] for LightGBM
        y = y + 1  # -1->0, 0->1, 1->2
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'learning_rate': self.config['training']['baseline']['learning_rate'],
                'num_leaves': self.config['training']['baseline']['num_leaves'],
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_data_in_leaf': 50,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(
                params, 
                train_data, 
                valid_sets=[val_data], 
                num_boost_round=self.config['training']['baseline']['num_boost_round'],
                callbacks=[lgb.early_stopping(self.config['training']['baseline']['early_stopping_rounds'])]
            )
            
            # Evaluate
            y_pred = np.argmax(model.predict(X_val), axis=1)  # Already in [0, 1, 2] range
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
        
        # Train final model on full dataset
        train_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, train_data, num_boost_round=self.config['training']['baseline']['num_boost_round'])
        
        # Save model and metadata
        model_name = f"lightgbm_sl_{symbol or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(self.models_dir, f"{model_name}.txt")
        final_model.save_model(model_path)
        
        metadata = {
            'feature_cols': feature_cols,
            'model_type': 'lightgbm',
            'symbol': symbol,
            'horizon': horizon,
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores),
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        joblib.dump(metadata, f"{model_path}.meta.pkl")
        
        print(f"LightGBM model saved: {model_path}")
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {np.mean(scores):.4f}")
        
        return {
            'model_path': model_path,
            'metadata': metadata,
            'cv_scores': scores
        }
    
    def train_pytorch_model(self, df: pd.DataFrame, symbol: str = None) -> Dict:
        """Train PyTorch CNN model."""
        horizon = self.config['labels']['horizon_minutes']
        label_col = f'label_{horizon}'
        window = self.config['dataset']['window']
        
        # Prepare dataset - save DataFrame to temporary CSV first
        temp_csv_path = os.path.join(self.models_dir, f"temp_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(temp_csv_path, index=False)
        
        dataset = TimeSeriesDataset(
            temp_csv_path, 
            label_col=label_col, 
            window=window,
            stride=self.config['dataset']['stride']
        )
        
        # Split data
        train_size = int(len(dataset) * (1 - self.config['dataset']['test_size'] - self.config['dataset']['validation_size']))
        val_size = int(len(dataset) * self.config['dataset']['validation_size'])
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['pytorch']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['pytorch']['batch_size'], shuffle=False)
        
        # Initialize model
        n_features = len(dataset.feature_cols)
        model = SmallCNN(n_features, window, n_classes=3)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['pytorch']['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training']['pytorch']['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, (batch_y + 1).long())  # Convert -1,0,1 to 0,1,2
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, (batch_y + 1).long())
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_name = f"pytorch_sl_{symbol or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = os.path.join(self.models_dir, f"{model_name}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'n_features': n_features,
                    'window': window,
                    'feature_cols': dataset.feature_cols,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['pytorch']['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model for final evaluation
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['pytorch']['batch_size'], shuffle=False)
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, (batch_y + 1).long())
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == (batch_y + 1).long()).sum().item()
        
        test_accuracy = correct / total
        
        metadata = {
            'feature_cols': dataset.feature_cols,
            'model_type': 'pytorch',
            'symbol': symbol,
            'horizon': horizon,
            'window': window,
            'test_accuracy': test_accuracy,
            'best_val_loss': best_val_loss,
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        joblib.dump(metadata, f"{model_path}.meta.pkl")
        
        # Clean up temporary file
        try:
            os.remove(temp_csv_path)
        except:
            pass
        
        print(f"PyTorch model saved: {model_path}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        return {
            'model_path': model_path,
            'metadata': metadata,
            'test_accuracy': test_accuracy
        }
    
    def train_all_models(self, symbols: List[str] = None) -> Dict:
        """Train both LightGBM and PyTorch models."""
        print("Preparing training data...")
        df = self.prepare_training_data(symbols)
        
        results = {}
        
        print("Training LightGBM model...")
        results['lightgbm'] = self.train_lightgbm_model(df)
        
        print("Training PyTorch model...")
        results['pytorch'] = self.train_pytorch_model(df)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Train supervised learning models")
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument("--symbols", nargs="+", help="Symbols to train on")
    parser.add_argument("--model_type", choices=["lightgbm", "pytorch", "all"], default="all")
    
    args = parser.parse_args()
    
    pipeline = SLTrainingPipeline(args.config)
    
    if args.model_type == "all":
        results = pipeline.train_all_models(args.symbols)
    elif args.model_type == "lightgbm":
        df = pipeline.prepare_training_data(args.symbols)
        results = {"lightgbm": pipeline.train_lightgbm_model(df)}
    elif args.model_type == "pytorch":
        df = pipeline.prepare_training_data(args.symbols)
        results = {"pytorch": pipeline.train_pytorch_model(df)}
    
    print("Training completed!")
    print("Results:", results)


if __name__ == "__main__":
    main()

