import requests
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

TOKEN_FILE = "tokens.json"

def login_to_tastytrade():
    email = os.getenv("TASTY_EMAIL")
    password = os.getenv("TASTY_PASSWORD")
    url = "https://api.tastyworks.com/sessions"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "login": email,
        "password": password
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        print("✅ Logged in to Tastytrade. [%s]" % (json.dumps(response.json())))
        data = response.json()['data']
        session_token = data['session-token']
        session_expires_at = data['session-expiration']  # Get the expiry timestamp
        return {
            "session_token": session_token,
            "expires_at": session_expires_at
        }
    else:
        print(f"❌ Login failed: {response.status_code} - {response.text}")
        return None



def save_tokens(session_data):
    token_data = {
        "tastytrade_session_token": session_data["session_token"],
        "tastytrade_expires_at": session_data["expires_at"]
    }

    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"💾Session Token saved to {TOKEN_FILE}")


def main():
    session_data = login_to_tastytrade()
    if not session_data:
        return

    save_tokens(session_data)


if __name__ == "__main__":
    main()