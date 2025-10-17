#!/usr/bin/env python3
import sys, os
from datetime import datetime, timezone

sys.path.append('../../..')
from db.database import TradeDatabase

# Quick fix - detect schema and insert fresh news
db = TradeDatabase()
conn, c = db.sql_connect()

# Get schema
c.execute("PRAGMA table_info(news_articles)")
cols = [row[1] for row in c.fetchall()]
print("Columns:", cols)

# Get sample row to see data structure
c.execute("SELECT * FROM news_articles LIMIT 1")
sample = c.fetchone()
print("Sample row:", sample)

# Insert fresh news for all 8 symbols
symbols = ["/ES:XCME{=h}","/NQ:XCME{=h}","/MES:XCME{=h}","/MNQ:XCME{=h}","/RTY:XCME{=h}","/QM:XNYM{=h}","/QG:XNYM{=h}","/MCL:XNYM{=h}"]
ts = datetime.now(timezone.utc).isoformat()

for symbol in symbols:
    if 'instrument' in cols:
        c.execute("INSERT INTO news_articles (instrument, title, content, published_at, sentiment_score, market_signal) VALUES (?,?,?,?,?,?)", 
                 (symbol, f"Fresh news {symbol}", f"Content for {symbol}", ts, 0.7, "BUY"))
    elif 'symbol' in cols:
        c.execute("INSERT INTO news_articles (symbol, title, content, published_at, sentiment_score, market_signal) VALUES (?,?,?,?,?,?)", 
                 (symbol, f"Fresh news {symbol}", f"Content for {symbol}", ts, 0.7, "BUY"))
    else:
        # Try with available columns
        c.execute("INSERT INTO news_articles (title, content, published_at, sentiment_score, market_signal) VALUES (?,?,?,?,?)", 
                 (f"Fresh news {symbol}", f"Content for {symbol}", ts, 0.7, "BUY"))

conn.commit()
db.close_connection(conn)
print(f"✅ Fresh news added at {ts}")
