import requests
import json
import os
import sys
from uuid import uuid4

# Import your database class from the other file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase

# Define the path to your token file
TOKEN_FILE_PATH = 'tokens.json'

def get_token_from_file():
    """Reads the session token from the JSON file."""
    if not os.path.exists(TOKEN_FILE_PATH):
        print(f"❌ Error: Token file not found at '{TOKEN_FILE_PATH}'")
        return None
    try:
        with open(TOKEN_FILE_PATH, 'r') as f:
            data = json.load(f)
            token = data.get('tastytrade_session_token')
            if not token:
                print("❌ Error: 'tastytrade_session_token' key not found in tokens.json.")
            return token
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from '{TOKEN_FILE_PATH}'.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while reading the token file: {e}")
        return None

# This database function remains unchanged
def update_positions_in_db(positions_data):
    """Connects to the database, clears the old positions, and inserts the new ones."""
    print("\nUpdating positions in the database...")
    conn, cursor = TradeDatabase.sql_connect()
    if not conn:
        print("❌ Could not connect to the database. Aborting update.")
        return

    try:
        cursor.execute("DELETE FROM positions")
        print("- Cleared old positions from the table.")
        position_items = positions_data.get("data", {}).get("items", [])
        if not position_items:
            print("- No new positions to add.")
            return
            
        positions_to_insert = [
            (
                str(uuid4()), item.get('account-number'), item.get('instrument-type'), item.get('symbol'),
                item.get('underlying-symbol'), item.get('quantity'), item.get('average-daily-market-close-price'),
                item.get('average-open-price'), item.get('average-yearly-market-close-price'),
                item.get('close-price'), item.get('cost-effect'), 1 if item.get('is-frozen') else 0,
                1 if item.get('is-suppressed') else 0, item.get('multiplier'), item.get('quantity-direction'),
                item.get('restricted-quantity'), item.get('expires-at'), item.get('realized-day-gain'),
                item.get('realized-day-gain-date'), item.get('realized-day-gain-effect'),
                item.get('realized-today'), item.get('realized-today-date'), item.get('realized-today-effect'),
                item.get('created-at'), item.get('updated-at')
            ) for item in position_items
        ]

        sql_insert_query = """
        INSERT INTO positions (id, account_number, instrument_type, symbol, underlying_symbol, quantity,
        average_daily_market_close_price, average_open_price, average_yearly_market_close_price,
        close_price, cost_effect, is_frozen, is_suppressed, multiplier, quantity_direction,
        restricted_quantity, expires_at, realized_day_gain, realized_day_gain_date,
        realized_day_gain_effect, realized_today, realized_today_date, realized_today_effect,
        created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        cursor.executemany(sql_insert_query, positions_to_insert)
        conn.commit()
        print(f"✅ Successfully inserted {cursor.rowcount} new positions into the database.")
    except Exception as e:
        print(f"❌ An error occurred during database update: {e}")
        conn.rollback()
    finally:
        TradeDatabase.close_connection(conn)


# --- Main Script Execution Flow ---
if __name__ == "__main__":
    # Ensure the database and tables exist before running
    TradeDatabase.create_tables()

    # Step 1: Get the session token directly from the file
    session_token = get_token_from_file()
    
    # Step 2: If a token exists, fetch and store positions
    if session_token:
        ACCOUNT_NUMBER = "5WU55726"
        POSITIONS_URL = f"https://api.tastyworks.com/accounts/{ACCOUNT_NUMBER}/positions"
        api_headers = {"Authorization": session_token}
        
        print(f"\nUsing token from file to fetch positions for account {ACCOUNT_NUMBER}...")
        try:
            positions_response = requests.get(POSITIONS_URL, headers=api_headers)
            # Raise an error for bad responses (like 401 Unauthorized if the token is bad)
            positions_response.raise_for_status() 
            
            positions_data = positions_response.json()
            print("✅ Successfully fetched positions!")
            
            update_positions_in_db(positions_data)

        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred while fetching positions: {e}")
            if e.response:
                print(f"Status Code: {e.response.status_code} (Note: 401 means your token is invalid or expired)")
                print(f"Response Body: {e.response.text}")
    else:
        print("Could not obtain a session token from tokens.json. Exiting.")