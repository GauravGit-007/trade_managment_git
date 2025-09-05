import os
import sys
from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
npNaN = np.nan

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "BTC/USD:CXTALP",
    "ETH/USD:CXTALP",
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
    timesteps = int(os.environ.get("RL_TIMESTEPS", "50000"))

    run_symbols = [single_symbol] if single_symbol else SYMBOLS

    for symbol in run_symbols:
        print(f"\n[TRAIN] Starting PPO training for {symbol}")

        start_time = datetime.now(timezone.utc)  # <-- mark training start

        env = DummyVecEnv([lambda s=symbol: make_env(s)])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)

        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        sym_tag = sanitize_symbol(canonicalize_symbol_for_db(symbol))
        policy_version = f"ppo_{sym_tag}_{ts_tag}"

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

        end_time = datetime.now(timezone.utc)  # <-- mark training end

        # Use canonicalized symbol for logging
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

'''