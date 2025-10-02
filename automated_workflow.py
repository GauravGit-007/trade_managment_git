#!/usr/bin/env python3
"""
Common Setup Script...

Automates end-to-end setup steps shared by RL and SL until monitor_actuals execution.
- Historical data
- News service
- Auth tokens
- Fetch positions
- Models: news model, LSTM (skipped if missing)
- Monitor actuals

Usage (PowerShell):
  python common_setup.py
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(script_rel_path: str, args=None):
    path = os.path.join(ROOT, script_rel_path)
    if not os.path.exists(path):
        print(f"[skip] Not found: {script_rel_path}")
        return 0

    cmd = [sys.executable, path]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running: {' '.join(cmd)} (cwd={ROOT})")
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"[warn] {script_rel_path} exited with code {result.returncode}")
    return result.returncode


def main():
    # Services
    run(os.path.join("services", "historical_data.py"))   #works fine ,requires the 9-10 min time and output will only be shown after all that time,at a single time
    run(os.path.join("services", "news_service.py"))     #works fine
    run(os.path.join("services", "auth_tokens.py"))      #works fine
    run(os.path.join("services", "fetch_positions.py"))  #works fine

    # Models (skip if not present)
    run(os.path.join("models", "news_model.py"))         #works fine
         
    #run(os.path.join("models", "lstm.py"))          #half works,need futher debugging,so no for now

    # Monitor actuals
    #run(os.path.join("services", "monitor_actuals.py"))    #works fine


if __name__ == "__main__":
    main()
