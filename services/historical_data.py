from dotenv import load_dotenv
import os, requests, sys
import asyncio
import websockets
import json
import sqlite3
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
def append_to_db(row):
    if all((isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.lower() == "nan") for x in row):
        print(f"⏩ Skipping row: all values are NaN: {row}")
        return

    conn, cursor = TradeDatabase.sql_connect()
    try:
        # It's good practice to store minute-data in a separate table
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

        row_with_uuid = [str(uuid.uuid4())] + row
        cursor.execute(f'''
            INSERT OR IGNORE INTO {table_name} (id, symbol, open, close, high, low, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', row_with_uuid)
        conn.commit()
    finally:
        TradeDatabase.close_connection(conn)

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

async def connect_to_dxlink(url: str, token: str, symbol: str, start_timestamp: int):
    async with websockets.connect(url) as ws:
        print("🔗 Connected to WebSocket")

        # Steps 1-4: SETUP, AUTH, CHANNEL_REQUEST, FEED_SETUP (remain the same)
        await ws.send(json.dumps({"type": "SETUP", "channel": 0, "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60, "version": "0.1-DXF-JS/0.3.0"}))
        await ws.send(json.dumps({"type": "AUTH", "channel": 0, "token": token}))
        await ws.send(json.dumps({"type": "CHANNEL_REQUEST", "channel": 1, "service": "FEED", "parameters": {"contract": "AUTO"}}))
        await ws.send(json.dumps({"type": "FEED_SETUP", "channel": 1, "acceptAggregationPeriod": 0, "acceptDataFormat": "COMPACT", "acceptEventFields": {"Candle": ["eventSymbol", "open", "close", "high", "low", "volume", "time"]}}))
        
        print("✅ Sent initial SETUP messages")

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

        print(f"✅ Sent FEED_SUBSCRIPTION for {symbol} (1-hour candles) from {datetime.fromtimestamp(start_timestamp / 1000, tz=timezone.utc)}")

        try:
            while True:
                response = await ws.recv()
                data = json.loads(response)

                if data.get("type") == "FEED_DATA" and "data" in data:
                    feed_type = data["data"][0]
                    feed_content = data["data"][1]

                    if feed_type == "Candle" and isinstance(feed_content, list):
                        candle_count = 0
                        for candle in parse_flat_candles(feed_content):
                            append_to_db(candle)
                            candle_count += 1
                        print(f"📥 {candle_count} 1-hour candle(s) saved to DB for {symbol}.")
        except websockets.ConnectionClosed:
            print("🔌 Connection closed.")

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

if __name__ == "__main__":
    symbols = [
        "/NQ:XCME", "/ES:XCME", "/RTY:XCME", "/QG:XNYM", "/QM:XNYM",
        "BTC/USD:CXTALP", "ETH/USD:CXTALP", "/MES:XCME", "/MNQ:XCME", "/MCL:XNYM"
    ]

    # Timestamp exactly 24 hours ago
    start_timestamp = get_last_24_hours_timestamps()

    session_token = login_to_tastyworks(email, password)
    if session_token:
        token, dxlink_url = get_api_quote_token(session_token)
        if token and dxlink_url:
            print("✅ Token acquired. Fetching last 24 hourly candles.")
            for symbol in symbols:
                print(f"\n📊 Fetching 1-hour candles for symbol: {symbol}")
                try:
                    asyncio.run(connect_to_dxlink(dxlink_url, token, symbol, start_timestamp))
                except Exception as e:
                    print(f"⚠️ Error fetching data for {symbol}: {e}")