"""
SL Decision Monitoring Service
Monitors SL model decisions, tracks performance, and provides alerts.
Mirrors the functionality of RL monitoring but for supervised learning models.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase
from supervised.infer_supervised import SupervisedInference
import joblib


@dataclass
class SLMonitoringConfig:
    """Configuration for SL monitoring service."""
    check_interval_minutes: int = 5
    performance_window_hours: int = 24
    alert_thresholds: Dict = None
    model_path: str = None
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'max_drawdown': 0.05,  # 5% max drawdown
                'min_win_rate': 0.4,   # 40% minimum win rate
                'max_loss_streak': 5,   # Max consecutive losses
                'min_confidence': 0.6   # Minimum prediction confidence
            }
        
        if self.symbols is None:
            self.symbols = [
                "/NQ:XCME", "/ES:XCME", "/RTY:XCME", "/QG:XNYM", "/QM:XNYM",
                "BTC/USD:CXTALP", "ETH/USD:CXTALP", "/MES:XCME", "/MNQ:XCME", "/MCL:XNYM"
            ]


class SLDecisionMonitor:
    """Monitors SL model decisions and performance."""
    
    def __init__(self, config: SLMonitoringConfig):
        self.config = config
        self.model = None
        self.model_metadata = None
        self._load_model()
        
    def _load_model(self):
        """Load the SL model for monitoring."""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        
        metadata_path = f"{self.config.model_path}.meta.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        self.model_metadata = joblib.load(metadata_path)
        model_type = self.model_metadata.get('model_type', 'lightgbm')
        
        self.model = SupervisedInference(
            self.config.model_path, 
            model_type, 
            self.model_metadata.get('feature_cols')
        )
        
        print(f"Loaded SL model: {self.config.model_path}")
        print(f"Model type: {model_type}")
        print(f"Features: {len(self.model_metadata.get('feature_cols', []))}")
    
    def get_recent_decisions(self, hours: int = 24) -> pd.DataFrame:
        """Get recent SL decisions from database."""
        conn, cursor = TradeDatabase.sql_connect()
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        query = """
        SELECT id, symbol, decision_timestamp, action, position_before, position_after,
               price, pnl_change, model_version, confidence, comment
        FROM sl_decisions
        WHERE decision_timestamp >= ?
        ORDER BY decision_timestamp DESC
        """
        
        cursor.execute(query, (cutoff_time.isoformat(),))
        rows = cursor.fetchall()
        TradeDatabase.close_connection(conn)
        
        if not rows:
            return pd.DataFrame()
        
        cols = ['id', 'symbol', 'decision_timestamp', 'action', 'position_before', 
                'position_after', 'price', 'pnl_change', 'model_version', 'confidence', 'comment']
        
        df = pd.DataFrame(rows, columns=cols)
        df['decision_timestamp'] = pd.to_datetime(df['decision_timestamp'])
        
        return df
    
    def calculate_performance_metrics(self, decisions_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from recent decisions."""
        if decisions_df.empty:
            return {
                'total_decisions': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_confidence': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'loss_streak': 0
            }
        
        # Basic metrics
        total_decisions = len(decisions_df)
        total_pnl = decisions_df['pnl_change'].sum()
        avg_confidence = decisions_df['confidence'].mean()
        
        # Win rate
        winning_decisions = decisions_df[decisions_df['pnl_change'] > 0]
        win_rate = len(winning_decisions) / total_decisions if total_decisions > 0 else 0.0
        
        # Drawdown calculation
        cumulative_pnl = decisions_df['pnl_change'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified)
        if decisions_df['pnl_change'].std() > 0:
            sharpe_ratio = decisions_df['pnl_change'].mean() / decisions_df['pnl_change'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Loss streak
        loss_streak = 0
        current_streak = 0
        for pnl in decisions_df['pnl_change'].iloc[::-1]:  # Check from most recent
            if pnl < 0:
                current_streak += 1
                loss_streak = max(loss_streak, current_streak)
            else:
                break
        
        return {
            'total_decisions': total_decisions,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'loss_streak': loss_streak
        }
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check if any alert conditions are met."""
        alerts = []
        thresholds = self.config.alert_thresholds
        
        # Check drawdown
        if abs(metrics['max_drawdown']) > thresholds['max_drawdown']:
            alerts.append({
                'type': 'high_drawdown',
                'severity': 'high',
                'message': f"High drawdown detected: {metrics['max_drawdown']:.2%} (threshold: {thresholds['max_drawdown']:.2%})",
                'value': metrics['max_drawdown'],
                'threshold': thresholds['max_drawdown']
            })
        
        # Check win rate
        if metrics['win_rate'] < thresholds['min_win_rate']:
            alerts.append({
                'type': 'low_win_rate',
                'severity': 'medium',
                'message': f"Low win rate: {metrics['win_rate']:.2%} (threshold: {thresholds['min_win_rate']:.2%})",
                'value': metrics['win_rate'],
                'threshold': thresholds['min_win_rate']
            })
        
        # Check loss streak
        if metrics['loss_streak'] >= thresholds['max_loss_streak']:
            alerts.append({
                'type': 'loss_streak',
                'severity': 'high',
                'message': f"Loss streak: {metrics['loss_streak']} consecutive losses (threshold: {thresholds['max_loss_streak']})",
                'value': metrics['loss_streak'],
                'threshold': thresholds['max_loss_streak']
            })
        
        # Check confidence
        if metrics['avg_confidence'] < thresholds['min_confidence']:
            alerts.append({
                'type': 'low_confidence',
                'severity': 'low',
                'message': f"Low average confidence: {metrics['avg_confidence']:.2f} (threshold: {thresholds['min_confidence']:.2f})",
                'value': metrics['avg_confidence'],
                'threshold': thresholds['min_confidence']
            })
        
        return alerts
    
    def log_alert(self, alert: Dict):
        """Log alert to database."""
        conn, cursor = TradeDatabase.sql_connect()
        
        cursor.execute("""
            INSERT INTO sl_alerts (
                alert_type, severity, message, alert_timestamp, 
                value, threshold, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert['type'],
            alert['severity'],
            alert['message'],
            datetime.now(timezone.utc).isoformat(),
            alert['value'],
            alert['threshold'],
            self.model_metadata.get('model_version', 'unknown')
        ))
        
        conn.commit()
        TradeDatabase.close_connection(conn)
    
    def get_model_health(self) -> Dict:
        """Get overall model health status."""
        recent_decisions = self.get_recent_decisions(self.config.performance_window_hours)
        metrics = self.calculate_performance_metrics(recent_decisions)
        alerts = self.check_alerts(metrics)
        
        # Determine overall health
        high_severity_alerts = [a for a in alerts if a['severity'] == 'high']
        medium_severity_alerts = [a for a in alerts if a['severity'] == 'medium']
        
        if high_severity_alerts:
            health_status = 'critical'
        elif medium_severity_alerts:
            health_status = 'warning'
        elif alerts:
            health_status = 'caution'
        else:
            health_status = 'healthy'
        
        return {
            'status': health_status,
            'metrics': metrics,
            'alerts': alerts,
            'model_info': {
                'model_path': self.config.model_path,
                'model_type': self.model_metadata.get('model_type'),
                'training_date': self.model_metadata.get('training_date'),
                'feature_count': len(self.model_metadata.get('feature_cols', []))
            },
            'last_check': datetime.now(timezone.utc).isoformat()
        }
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        print(f"[SL Monitor] Running monitoring cycle at {datetime.now()}")
        
        health = self.get_model_health()
        
        # Log any new alerts
        for alert in health['alerts']:
            self.log_alert(alert)
            print(f"[ALERT] {alert['severity'].upper()}: {alert['message']}")
        
        # Print summary
        metrics = health['metrics']
        print(f"[SL Monitor] Status: {health['status'].upper()}")
        print(f"[SL Monitor] Decisions: {metrics['total_decisions']}, PnL: ${metrics['total_pnl']:.2f}")
        print(f"[SL Monitor] Win Rate: {metrics['win_rate']:.2%}, Confidence: {metrics['avg_confidence']:.2f}")
        
        return health
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        import time
        
        print(f"[SL Monitor] Starting continuous monitoring...")
        print(f"[SL Monitor] Check interval: {self.config.check_interval_minutes} minutes")
        print(f"[SL Monitor] Performance window: {self.config.performance_window_hours} hours")
        
        try:
            while True:
                self.run_monitoring_cycle()
                time.sleep(self.config.check_interval_minutes * 60)
        except KeyboardInterrupt:
            print("[SL Monitor] Monitoring stopped by user")
        except Exception as e:
            print(f"[SL Monitor] Error: {e}")


def create_sl_alerts_table():
    """Create the sl_alerts table if it doesn't exist."""
    conn, cursor = TradeDatabase.sql_connect()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sl_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            alert_timestamp TEXT NOT NULL,
            value REAL,
            threshold REAL,
            model_version TEXT,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_timestamp TEXT
        )
    """)
    
    conn.commit()
    TradeDatabase.close_connection(conn)
    print("Created sl_alerts table")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SL Decision Monitoring Service")
    parser.add_argument("--model", required=True, help="Path to SL model file")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in minutes")
    parser.add_argument("--window", type=int, default=24, help="Performance window in hours")
    parser.add_argument("--once", action="store_true", help="Run once instead of continuously")
    parser.add_argument("--init-db", action="store_true", help="Initialize database tables")
    
    args = parser.parse_args()
    
    if args.init_db:
        create_sl_alerts_table()
        return
    
    config = SLMonitoringConfig(
        model_path=args.model,
        check_interval_minutes=args.interval,
        performance_window_hours=args.window
    )
    
    monitor = SLDecisionMonitor(config)
    
    if args.once:
        health = monitor.run_monitoring_cycle()
        print("\nModel Health Summary:")
        print(json.dumps(health, indent=2, default=str))
    else:
        monitor.start_monitoring()


if __name__ == "__main__":
    main()
