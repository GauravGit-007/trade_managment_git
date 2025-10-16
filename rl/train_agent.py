import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
npNaN = np.nan

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


import json
#from uuid import uuid4
#from datetime import datetime, timezone
from stable_baselines3.common.callbacks import BaseCallback
from db.database import TradeDatabase

def convert_np_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_np_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_list(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class RLExperienceLoggerCallback(BaseCallback):
    def __init__(self, symbol, policy_version, verbose=0):
        super().__init__(verbose)
        self.symbol = symbol
        self.policy_version = policy_version
        self.episode_id = str(uuid4())

    def serialize_state(self, state):
        if isinstance(state, np.ndarray):
            return json.dumps(state.tolist())
        if hasattr(state, "tolist"):
            return json.dumps(state.tolist())
        try:
            return json.dumps(state)
        except TypeError:
        # If all else fails, convert to string
            return json.dumps(str(state))

    def _on_step(self) -> bool:
        try:
            obs = self.locals.get('obs')
            actions = self.locals.get('actions')
            rewards = self.locals.get('rewards')
            dones = self.locals.get('dones')
            infos = self.locals.get('infos')

            for i in range(len(rewards)):
                state = obs[i] if obs is not None else None
                action = actions[i] if actions is not None else None
                reward = rewards[i] if rewards is not None else None
                done = dones[i] if dones is not None else None
                info = infos[i] if infos is not None else None

                conn, cursor = TradeDatabase.sql_connect()
                cursor.execute(
                    """
                    INSERT INTO rl_experiences (
                        id, symbol, t_timestamp, state_json, action, reward, next_state_json,
                        done, episode_id, info_json, policy_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        self.symbol,
                        datetime.now(timezone.utc).isoformat(),
                        self.serialize_state(state),
                        int(action) if action is not None else None,
                        float(reward) if reward is not None else None,
                        None,  # next_state not available here
                        int(done) if done is not None else None,
                        self.episode_id,
                        json.dumps(convert_np_to_list(info)) if info is not None else None,

                        self.policy_version
                    )
                )
                conn.commit()
                TradeDatabase.close_connection(conn)

                # Reset episode_id at episode end
                if done:
                    self.episode_id = str(uuid4())

        except Exception as e:
            if self.verbose > 0:
                print(f"Error logging experience: {e}")

        return True






from rl.env import TradingEnv, EnvConfig
from db.database import TradeDatabase


def log_episode_summary(symbol: str, total_reward: float, steps: int, policy_version: str,
                        start_time: datetime, end_time: datetime):
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute(
        """
        INSERT INTO rl_episodes (
            id, symbol, start_timestamp, end_timestamp,
            total_reward, steps, policy_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            symbol,
            start_time.isoformat(),
            end_time.isoformat(),
            float(total_reward),
            int(steps),
            policy_version,
        ),
    )
    conn.commit()
    TradeDatabase.close_connection(conn)



SYMBOLS = [
    "/NQ:XCME",
    "/ES:XCME",
    "/RTY:XCME",
    "/QG:XNYM",
    "/QM:XNYM",
    #"BTC/USD:CXTALP",
    #"ETH/USD:CXTALP",
    "/MES:XCME",
    "/MNQ:XCME",
    "/MCL:XNYM",
]


def canonicalize_symbol_for_db(symbol: str) -> str:
    return symbol if "{=" in symbol else f"{symbol}{{=h}}"


def sanitize_symbol(symbol: str) -> str:
    return (
        symbol.replace("/", "_")
        .replace(":", "-")
        .replace("{", "")
        .replace("}", "")
        .replace("=", "")
    )


def make_env(symbol: str) -> TradingEnv:
    db_symbol = canonicalize_symbol_for_db(symbol)
    cfg = EnvConfig(symbol=db_symbol)
    return TradingEnv(cfg)


def main():
    single_symbol = os.environ.get("RL_SYMBOL")
    #timesteps = int(os.environ.get("RL_TIMESTEPS", "50000"))
    #just quickly testing with 500 timesteps
    timesteps = int(os.environ.get("RL_TIMESTEPS", "500000"))
    run_symbols = [single_symbol] if single_symbol else SYMBOLS

    for symbol in run_symbols:
        print(f"\n[TRAIN] Starting PPO training for {symbol}")

        start_time = datetime.now(timezone.utc)  # <-- mark training start

        env = DummyVecEnv([lambda s=symbol: make_env(s)])
        model = PPO("MlpPolicy", env, verbose=1)

        ts_tag = start_time.strftime("%Y%m%d%H%M%S")
        sym_tag = sanitize_symbol(canonicalize_symbol_for_db(symbol))
        policy_version = f"ppo_{sym_tag}_{ts_tag}"

        # Initialize callback with policy version & symbol
        callback = RLExperienceLoggerCallback(
            symbol=canonicalize_symbol_for_db(symbol),
            policy_version=policy_version,
            verbose=1
        )

        # Train with callback to log experiences
        model.learn(total_timesteps=timesteps, callback=callback)

        end_time = datetime.now(timezone.utc)  # <-- mark training end

        model_path = os.path.join("outputs", f"{policy_version}.zip")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        # Evaluate briefly
        eval_env = make_env(symbol)
        obs, _ = eval_env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(int(action))
            total_reward += float(reward)
            steps += 1

        # Log episode summary with duration
        log_episode_summary(
            canonicalize_symbol_for_db(symbol),
            total_reward,
            steps,
            policy_version,
            start_time,
            end_time,
        )

        print(
            f"[TRAIN] Saved model to {model_path}. "
            f"Eval reward={total_reward:.2f} steps={steps} "
            f"(duration={(end_time - start_time).total_seconds():.1f}s)"
        )




if __name__ == "__main__":
    main()
'''

import os
import sys
from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env import TradingEnv, EnvConfig
from db.database import TradeDatabase


def log_episode_summary(symbol: str, total_reward: float, steps: int, policy_version: str):
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute(
        """
        INSERT INTO rl_episodes (id, symbol, start_timestamp, end_timestamp, total_reward, steps, policy_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            symbol,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            float(total_reward),
            int(steps),
            policy_version,
        ),
    )
    conn.commit()
    TradeDatabase.close_connection(conn)


def make_env(symbol: str) -> TradingEnv:
    cfg = EnvConfig(symbol=symbol)
    return TradingEnv(cfg)


def main():
    symbol = os.environ.get("RL_SYMBOL", "/ES:XCME{=h}")
    timesteps = int(os.environ.get("RL_TIMESTEPS", "50000"))

    env = DummyVecEnv([lambda: make_env(symbol)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    policy_version = f"ppo_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    model_path = os.path.join("outputs", f"{policy_version}.zip")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Evaluate briefly
    eval_env = make_env(symbol)
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(int(action))
        total_reward += float(reward)
        steps += 1

    log_episode_summary(symbol, total_reward, steps, policy_version)
    print(f"Saved model to {model_path}. Eval reward={total_reward:.2f} steps={steps}")



if __name__ == "__main__":
    main()

'''                        #CORRECT TRAIN_AGENT.PY CODE