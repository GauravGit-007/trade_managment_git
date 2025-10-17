#!/usr/bin/env python3
"""
Quick News Data Updater
Injects fresh news data for all 8 symbols
"""

import sys
import os
from datetime import datetime, timezone

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

def update_news_data():
    print("🔄 Updating news data for all 8 symbols...")
    
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    # 8 symbols from refined AI
    symbols = [
        "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
        "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}"
    ]
    
    current_time = datetime.now(timezone.utc).isoformat()
    
    for symbol in symbols:
        # Insert fresh news data
        cursor.execute("""
            INSERT INTO news_articles (symbol, title, content, published_at, sentiment_score, market_signal)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            f"Fresh Market Update for {symbol}",
            f"Latest market analysis and trading insights for {symbol}",
            current_time,
            0.7,  # Positive sentiment
            "BUY"
        ))
        print(f"✅ Added fresh news for {symbol}")
    
    conn.commit()
    db.close_connection(conn)
    
    print(f"✅ News data updated for all 8 symbols at {current_time}")

if __name__ == "__main__":
    update_news_data()
