import json
from uuid import uuid4
from datetime import datetime, timezone
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import TradeDatabase

# Your logging function and test code follow...

from db.database import TradeDatabase

def serialize_state(state):
    if hasattr(state, "tolist"):
        return json.dumps(state.tolist())
    return json.dumps(state)
print("Starting test of log_experience...")
def log_experience(symbol, t_timestamp, state, action, reward, next_state,
                   done, episode_id=None, info=None, position_before=None,
                   position_after=None, executed_delta=None, price_prev=None,
                   price_next=None, transaction_cost=None, risk_penalty=None,
                   cash_pnl=None, policy_version=None, decision_id=None):
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute(
        """
        INSERT INTO rl_experiences (
            id, symbol, t_timestamp, state_json, action, reward, next_state_json, 
            done, episode_id, info_json, position_before, position_after, executed_delta, 
            price_prev, price_next, transaction_cost, risk_penalty, cash_pnl,
            policy_version, decision_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()), symbol, t_timestamp.isoformat(),
            serialize_state(state), int(action), float(reward),
            serialize_state(next_state), int(done),
            episode_id,
            json.dumps(info) if info is not None else None,
            position_before, position_after, executed_delta,
            price_prev, price_next, transaction_cost, risk_penalty, cash_pnl,
            policy_version, decision_id,
        )
    )
    conn.commit()
    TradeDatabase.close_connection(conn)

print("Log experience function called successfully.")