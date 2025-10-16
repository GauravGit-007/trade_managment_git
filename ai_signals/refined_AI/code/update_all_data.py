#!/usr/bin/env python3
"""
Unified updater for Refined AI data prerequisites:
- News scraping (Yahoo) + instrument mapping via Azure OpenAI
- News sentiment (FinBERT) → writes to sentiment_analysis and JSON for LLM
- Historical 1h candles (TastyWorks/DXLink) → writes to historical_data_1h
- LSTM predictions (next 12h) → writes to lstm_predictions

Usage examples:
  python update_all_data.py --all
  python update_all_data.py --news
  python update_all_data.py --historical
  python update_all_data.py --lstm

Optional:
  --skip-news-sentiment  Only scrape news without running FinBERT model
  --quiet                Minimal logs
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure project root on path
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(ROOT_DIR)


def run_news_scrape(run_sentiment: bool = True, quiet: bool = False) -> bool:
    try:
        if not quiet:
            print("\n📰 Updating news (scrape + mapping)...")
        import services.news_service as news_service
        news = news_service.get_data()
        news_service.store_news_to_sqlite(news)
        if not quiet:
            print(f"✅ News scraped: {len(news)} articles")

        if run_sentiment:
            if not quiet:
                print("🤖 Running FinBERT sentiment on news...")
            import models.news_model as news_model
            news_model.run_pipeline()
            if not quiet:
                print("✅ Sentiment updated in DB and JSON exported")
        return True
    except Exception as e:
        print(f"❌ News update failed: {e}")
        return False


def run_historical_update(quiet: bool = False) -> bool:
    try:
        # The models/lstm.py now depends on historical_data_1h being present.
        # Historical fetcher resides in models/lstm.py (websocket pull utility embedded)
        if not quiet:
            print("\n📈 Updating historical 1h candles (last 24h)...")
        # Use the entrypoint in models/lstm.py that fetches last 24h 1h candles
        import importlib.util
        lstm_path = os.path.join(ROOT_DIR, "models", "lstm.py")
        spec = importlib.util.spec_from_file_location("lstm_mod", lstm_path)
        lstm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lstm_mod)  # type: ignore

        # Reuse its main block that fetches last 24h 1h candles
        start_ts = lstm_mod.get_last_24_hours_timestamps()
        email = os.getenv('email')
        password = os.getenv('password')
        session_token = lstm_mod.login_to_tastyworks(email, password)
        if session_token:
            token_dx = lstm_mod.get_api_quote_token(session_token)
            if token_dx:
                token, dxlink_url = token_dx
                if not quiet:
                    print("[OK] Token acquired. Fetching last 24 hourly candles.")
                symbols = [
                    "/NQ:XCME", "/ES:XCME", "/RTY:XCME", "/QG:XNYM", "/QM:XNYM",
                    "BTC/USD:CXTALP", "ETH/USD:CXTALP", "/MES:XCME", "/MNQ:XCME", "/MCL:XNYM"
                ]
                import asyncio
                for sym in symbols:
                    if not quiet:
                        print(f"  → {sym}")
                    try:
                        asyncio.run(lstm_mod.connect_to_dxlink(dxlink_url, token, sym, start_ts))
                    except Exception as e:
                        print(f"   ⚠️  {sym}: {e}")
        return True
    except Exception as e:
        print(f"❌ Historical update failed: {e}")
        return False


def run_lstm_predictions(quiet: bool = False) -> bool:
    try:
        if not quiet:
            print("\n🧠 Training LSTM and writing next-12h predictions...")
        # models.lstm.main() handles: load historical -> TA -> train -> predict -> upsert
        import models.lstm as lstm
        lstm.main()
        if not quiet:
            print("✅ LSTM predictions updated")
        return True
    except Exception as e:
        print(f"❌ LSTM update failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Refined AI unified data updater")
    parser.add_argument("--all", action="store_true", help="Run all updaters: news + sentiment, historical, LSTM")
    parser.add_argument("--news", action="store_true", help="Run news scraping + mapping (and sentiment unless skipped)")
    parser.add_argument("--skip-news-sentiment", action="store_true", help="Do not run FinBERT sentiment after scraping news")
    parser.add_argument("--historical", action="store_true", help="Update historical 1h candles (last 24h)")
    parser.add_argument("--lstm", action="store_true", help="Train LSTM and write next-12h predictions")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")

    args = parser.parse_args()

    if not any([args.all, args.news, args.historical, args.lstm]):
        parser.print_help()
        return 1

    start = datetime.now()
    if not args.quiet:
        print("🤖 Refined AI Data Updater")
        print(f"Started: {start.isoformat()}")
        print("=" * 60)

    ok = True

    if args.all or args.news:
        ok = run_news_scrape(run_sentiment=not args.skip_news_sentiment, quiet=args.quiet) and ok

    if args.all or args.historical:
        ok = run_historical_update(quiet=args.quiet) and ok

    if args.all or args.lstm:
        ok = run_lstm_predictions(quiet=args.quiet) and ok

    end = datetime.now()
    if not args.quiet:
        print("\n" + ("✅" if ok else "❌"), "Update flow finished.")
        print(f"Elapsed: {end - start}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())


