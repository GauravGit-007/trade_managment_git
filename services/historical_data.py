from dotenv import load_dotenv
import os, requests, sys
import asyncio
import websockets
import json
import sqlite3
import argparse
import time
import uuid
from datetime import datetime, timezone, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase
import math
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# load credentials from .env file
email = os.getenv('email')
password = os.getenv('password')


def login_to_tastyworks(email, password):
    try:
        url = "https://api.cert.tastyworks.com/sessions"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Cookie": "AWSALB=oPI/RDDUD2THwQQk1zuVEhu7KlrS8sQwAVog5eP08ezJtOC+3yLhAuXL2SJ+JV6Z51NrIB7P7fWJ83I7PuWT4glpBxnEE63+IDePUOeZptrtAwuqUA6Yfw/rHK8v; AWSALBCORS=oPI/RDDUD2THwQQk1zuVEhu7KlrS8sQwAVog5eP08ezJtOC+3yLhAuXL2SJ+JV6Z51NrIB7P7fWJ83I7PuWT4glpBxnEE63+IDePUOeZptrtAwuqUA6Yfw/rHK8v"
        }
        data = {
            "login": email,
            "password": password,
            "remember-me": True
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
            data = response.json()
            return data['data']['session-token']
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def get_api_quote_token(session_token):
    try:
        url = "https://api.cert.tastyworks.com/api-quote-tokens"
        headers = {
            "Authorization": session_token,
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            dxlink_url = data['data']['dxlink-url']
            token = data['data']['token']
            return token, dxlink_url
        else:
            print(f"Failed to get API quote token: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# --- MODIFICATION 1: Function to get a timestamp for the last 24 hours ---
def get_start_timestamp_for_1h_data() -> int:
    """Returns the Unix timestamp in milliseconds for 90 days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=1)
    return int(dt.timestamp()) * 1000

def get_last_24_hours_timestamps():
    """Return start timestamp for exactly 24 hourly candles."""
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)
    return int(start_time.timestamp()) * 1000

# --- MODIFICATION 2: Changed table name for clarity ---
def ensure_schema(cursor):
    table_name = "historical_data_1h"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume INTEGER,
            timestamp TEXT
        )
    ''')
    # Unique index for idempotent upserts per symbol/timestamp
    cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_symbol_time
        ON {table_name}(symbol, timestamp)
    """)
    # Query speed indexes
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)")


def batch_upsert_candles(cursor, rows):
    """Upsert many candles using the unique (symbol,timestamp) index.
    rows is a list of [symbol, open, close, high, low, volume, timestamp]
    """
    if not rows:
        return 0
    table_name = "historical_data_1h"
    # Build tuples for executemany; generate UUIDs per row
    payload = [(str(uuid.uuid4()), r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows if not (
        all((isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and str(x).lower() == "nan") for x in r)
    )]
    # Use INSERT OR REPLACE to deduplicate by unique index (symbol,timestamp)
    cursor.executemany(
        f"""
        INSERT OR REPLACE INTO {table_name}
        (id, symbol, open, close, high, low, volume, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def save_with_retry(cursor, rows, retries: int = 5, delay_sec: float = 0.5) -> int:
    """Retry wrapper around batch_upsert_candles to handle 'database is locked'."""
    attempt = 0
    while True:
        try:
            return batch_upsert_candles(cursor, rows)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < retries:
                time.sleep(delay_sec * (attempt + 1))
                attempt += 1
                continue
            raise

def parse_flat_candles(flat_data: list):
    chunk_size = 7
    for i in range(0, len(flat_data), chunk_size):
        chunk = flat_data[i:i + chunk_size]
        if len(chunk) == chunk_size:
            # Convert timestamp from milliseconds to ISO 8601 format
            chunk[6] = datetime.fromtimestamp(chunk[6] / 1000, tz=timezone.utc).isoformat()
            yield chunk

# This function may not be necessary for minute data but is kept for reference
def data_exists(symbol: str, date: str) -> bool:
    # This check might need adjustment depending on your logic for minute data
    return False

# This is no longer the primary date function we'll use
def get_yesterday_date() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

async def connect_to_dxlink(url: str, token: str, symbol: str, start_timestamp: int, cursor=None, commit_interval: int = 500):
    async with websockets.connect(url) as ws:
        print("[OK] Connected to WebSocket")

        # Steps 1-4: SETUP, AUTH, CHANNEL_REQUEST, FEED_SETUP (remain the same)
        await ws.send(json.dumps({"type": "SETUP", "channel": 0, "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60, "version": "0.1-DXF-JS/0.3.0"}))
        await ws.send(json.dumps({"type": "AUTH", "channel": 0, "token": token}))
        await ws.send(json.dumps({"type": "CHANNEL_REQUEST", "channel": 1, "service": "FEED", "parameters": {"contract": "AUTO"}}))
        await ws.send(json.dumps({"type": "FEED_SETUP", "channel": 1, "acceptAggregationPeriod": 0, "acceptDataFormat": "COMPACT", "acceptEventFields": {"Candle": ["eventSymbol", "open", "close", "high", "low", "volume", "time"]}}))
        
        print("[OK] Sent initial SETUP messages")

        # --- MODIFICATION 3: Change the FEED_SUBSCRIPTION message ---
        await ws.send(json.dumps({
            "type": "FEED_SUBSCRIPTION",
            "channel": 1,
            "reset": True,
            "add": [{
                "type": "Candle",
                # Change "{=1d}" for daily to "{=1m}" for minute data
                "symbol": f"{symbol}{{=1h}}",
                # Use the new start timestamp for fetching recent minute data
                "fromTime": start_timestamp
            }]
        }))

        print(f"[OK] Sent FEED_SUBSCRIPTION for {symbol} (1-hour candles) from {datetime.fromtimestamp(start_timestamp / 1000, tz=timezone.utc)}")

        try:
            buffer = []
            total_saved = 0
            while True:
                response = await ws.recv()
                data = json.loads(response)

                if data.get("type") == "FEED_DATA" and "data" in data:
                    feed_type = data["data"][0]
                    feed_content = data["data"][1]

                    if feed_type == "Candle" and isinstance(feed_content, list):
                        for candle in parse_flat_candles(feed_content):
                            # candle = [symbol, open, close, high, low, volume, timestamp]
                            buffer.append(candle)
                            if cursor is not None and len(buffer) >= commit_interval:
                                saved = save_with_retry(cursor, buffer)
                                buffer.clear()
                                total_saved += saved
                        if cursor is not None and buffer:
                            saved = save_with_retry(cursor, buffer)
                            total_saved += saved
                            buffer.clear()
                        print(f"[OK] {total_saved} 1-hour candle(s) saved to DB for {symbol}.")
        except websockets.ConnectionClosed:
            print("[OK] Connection closed.")

"""
if __name__ == "__main__":
    symbols = [
        "/NQ:XCME",
        "/ES:XCME",
        "/RTY:XCME",
        "/QG:XNYM",
        "/QM:XNYM",
        "BTC/USD:CXTALP",
        "ETH/USD:CXTALP",
        "/MES:XCME",
        "/MNQ:XCME",
        "/MCL:XNYM"
    ]

    session_token = login_to_tastyworks(email, password)
    if session_token:
        token, dxlink_url = get_api_quote_token(session_token)
        if token and dxlink_url:
            print(f"✅ Token acquired. Fetching 1-minute candle data for the last 7 days...")

            for symbol in symbols:
                print(f"\n📊 Fetching data for symbol: {symbol}")
                for days_back in range(1, 8):  # Fetch from 1 to 7 days ago
                    start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
                    start_timestamp = int(start_time.timestamp()) * 1000
                    print(f"➡️  Fetching data for {symbol} on {start_time.strftime('%Y-%m-%d')}")
                    try:
                        asyncio.run(connect_to_dxlink(dxlink_url, token, symbol, start_timestamp))
                    except Exception as e:
                        print(f"⚠️  Error fetching data for {symbol} on {start_time.strftime('%Y-%m-%d')}: {e}")

if __name__ == "__main__":
    symbols = [
        "/NQ:XCME", "/ES:XCME", "/RTY:XCME", "/QG:XNYM", "/QM:XNYM",
        "BTC/USD:CXTALP", "ETH/USD:CXTALP", "/MES:XCME", "/MNQ:XCME", "/MCL:XNYM"
    ]
    
    start_timestamp = get_start_timestamp_for_1h_data()

    session_token = login_to_tastyworks(email, password)
    if session_token:
        token, dxlink_url = get_api_quote_token(session_token)
        if token and dxlink_url:
            print("✅ Token acquired. Fetching 1-hour candle data for the last 1 days.")
            for symbol in symbols:
                print(f"\n📊 Fetching 1-hour candles for symbol: {symbol}")
                try:
                    asyncio.run(connect_to_dxlink(dxlink_url, token, symbol, start_timestamp))
                except Exception as e:
                    print(f"⚠️ Error fetching data for {symbol}: {e}")
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch 1-hour historical candles and store in SQLite")
    parser.add_argument("--symbols", nargs="*", default=[
        "/NQ:XCME", "/ES:XCME", "/RTY:XCME", "/QG:XNYM", "/QM:XNYM",
        "/MES:XCME", "/MNQ:XCME", "/MCL:XNYM"
    ], help="Symbols to fetch (root family without {=1h})")
    parser.add_argument("--hours", type=int, default=24, help="How many hours back to fetch")
    parser.add_argument("--commit-interval", type=int, default=500, help="Batch size for DB commits")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Timestamp N hours ago
    start_time = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    start_timestamp = int(start_time.timestamp()) * 1000

    session_token = login_to_tastyworks(email, password)
    if session_token:
        token, dxlink_url = get_api_quote_token(session_token)
        if token and dxlink_url:
            print(f"[OK] Token acquired. Fetching last {args.hours} hourly candles.")
            conn, cursor = TradeDatabase.sql_connect()
            try:
                ensure_schema(cursor)
                for symbol in args.symbols:
                    print(f"\n [OK] Fetching 1-hour candles for symbol: {symbol}")
                    try:
                        asyncio.run(connect_to_dxlink(dxlink_url, token, symbol, start_timestamp, cursor, args.commit_interval))
                        conn.commit()
                    except Exception as e:
                        print(f"[X] Error fetching data for {symbol}: {e}")
                conn.commit()
            finally:
                TradeDatabase.close_connection(conn)