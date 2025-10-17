# Refined AI Main Processor
# Generates daily AI signals for 8 symbols using historical data, news, and LSTM predictions

import os
import sys
import json
import re
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from statistics import mean
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from uuid import uuid4

# Add parent directories to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
root_dir = os.path.dirname(grandparent_dir)
sys.path.append(root_dir)
from db.database import TradeDatabase
import importlib

# Configure encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

class RefinedAIProcessor:
    def __init__(self, interactive_mode=True):
        """Initialize the Refined AI Processor"""
        self.client = AzureOpenAI(
            api_key="71b66107a84e489ea700ef4188d29947",
            azure_endpoint="https://vastai-openai-swedencentral.openai.azure.com/",
            api_version="2024-02-15-preview"
        )
        
        # 8 symbols (excluding Bitcoin and Ethereum)
        self.symbols = [
            "/ES:XCME{=h}",      # S&P 500 E-mini
            "/NQ:XCME{=h}",      # Nasdaq-100 E-mini
            "/MES:XCME{=h}",     # Micro S&P 500 E-mini
            "/MNQ:XCME{=h}",     # Micro Nasdaq-100 E-mini
            "/RTY:XCME{=h}",     # Russell 2000 E-mini
            "/QM:XNYM{=h}",      # Crude Oil E-mini
            "/QG:XNYM{=h}",      # Natural Gas E-mini
            "/MCL:XNYM{=h}"      # Micro Crude Oil
        ]
        
        self.interactive_mode = interactive_mode
        self.db = TradeDatabase()
        self.create_smart_ai_decisions_table()
        # Prefetched website news cache: { symbol: {latest_timestamp, hours_old, latest_url, available} }
        self.prefetched_news_map = {}

    def _format_hours(self, value):
        try:
            if value is None:
                return "-"
            return f"{float(value):.1f}h"
        except Exception:
            return "-"

    def _print_freshness_table(self, freshness_by_symbol: dict):
        """Render an interactive-style table with symbols as columns and 3 param rows."""
        symbols = self.symbols
        headers = ["param"] + symbols
        rows = []
        # Row 1: Historical hours old
        hist_row = ["Historical (h)"]
        for s in symbols:
            h = (freshness_by_symbol.get(s, {})
                 .get('historical_data', {})
                 .get('hours_old'))
            hist_row.append(self._format_hours(h))
        rows.append(hist_row)

        # Row 2: News (DB) hours old
        news_row = ["News DB (h)"]
        for s in symbols:
            h = (freshness_by_symbol.get(s, {})
                 .get('news_data', {})
                 .get('hours_old'))
            news_row.append(self._format_hours(h))
        rows.append(news_row)

        # Row 3: LSTM hours old
        lstm_row = ["LSTM (h)"]
        for s in symbols:
            h = (freshness_by_symbol.get(s, {})
                 .get('lstm_predictions', {})
                 .get('hours_old'))
            lstm_row.append(self._format_hours(h))
        rows.append(lstm_row)

        # Print table
        print("\n🧭 Data Freshness Overview (hours old)")
        # Compute column widths
        col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
        # Header
        header_line = " | ".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
        print(header_line)
        print("-" * len(header_line))
        # Rows
        for r in rows:
            print(" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))

    def prefetch_website_news(self):
        """Fetch website news once and compute latest article per symbol family."""
        try:
            news_service = importlib.import_module('services.news_service')
            articles = news_service.get_data()  # single scrape
        except Exception as e:
            # If scraping fails, leave map empty
            self.prefetched_news_map = {}
            return

        from datetime import timezone
        current_time = datetime.now(timezone.utc)

        result = {}
        for symbol in self.symbols:
            related = set(self.get_related_instruments(symbol))
            latest_ts = None
            latest_url = None
            for a in articles:
                inst = (a.get('instrument') or '').split(',')
                inst = {s.strip() for s in inst if s.strip()}
                if related & inst:
                    try:
                        ts = datetime.fromisoformat(a.get('published_at'))
                    except Exception:
                        continue
                    if latest_ts is None or ts > latest_ts:
                        latest_ts = ts
                        latest_url = a.get('url')
            if latest_ts is not None:
                hours_old = (current_time - latest_ts).total_seconds() / 3600
                result[symbol] = {
                    'available': True,
                    'latest_timestamp': latest_ts.isoformat(),
                    'hours_old': hours_old,
                    'latest_url': latest_url
                }
            else:
                result[symbol] = {
                    'available': False,
                    'latest_timestamp': None,
                    'hours_old': None,
                    'latest_url': None
                }

        self.prefetched_news_map = result

    def prompt_global_selection(self, freshness_by_symbol: dict) -> tuple[set, str]:
        """Ask user to proceed for all or select individual symbols based on freshness."""
        if not self.interactive_mode:
            # In non-interactive mode, default to only those with all_fresh
            return ({s for s, r in freshness_by_symbol.items() if r.get('all_fresh')}, 'fresh')

        # Show table first
        self._print_freshness_table(freshness_by_symbol)
        print("\n🗞️ Website freshness (latest article age and URL):")
        for s in self.symbols:
            web = freshness_by_symbol.get(s, {}).get('news_data_website', {})
            if web.get('available'):
                url_msg = f" | URL: {web.get('latest_url')}" if web.get('latest_url') else ""
                print(f"  {s}: {web.get('hours_old'):.1f}h{url_msg}")
            else:
                print(f"  {s}: -")

        print("\nProceed with generating signals:")
        print("  - Type 'all' to process all symbols (no further prompts)")
        print("  - Type comma-separated symbols to process subset; you will confirm each (y/n)")
        print("  - Press Enter to process only symbols with fresh data")

        try:
            choice = input("Your choice: ").strip()
        except Exception:
            choice = ""

        if choice.lower() == 'all':
            return (set(self.symbols), 'all')
        if not choice:
            return ({s for s, r in freshness_by_symbol.items() if r.get('all_fresh')}, 'fresh')

        selected = [x.strip() for x in choice.split(',') if x.strip()]
        valid = [s for s in selected if s in self.symbols]
        if not valid:
            print("No valid symbols provided; defaulting to only fresh symbols.")
            return ({s for s, r in freshness_by_symbol.items() if r.get('all_fresh')}, 'fresh')

        # Confirm each selected symbol
        confirmed = set()
        for s in valid:
            while True:
                try:
                    ans = input(f"Process {s}? (y/n): ").strip().lower()
                except Exception:
                    ans = 'n'
                if ans in ('y', 'yes'):
                    confirmed.add(s)
                    break
                if ans in ('n', 'no'):
                    break
                print("Please answer y or n.")
        return (confirmed, 'subset')

    def get_related_instruments(self, symbol):
        """Return a list of related instrument codes considered the same news family."""
        families = {
            "/ES:XCME{=h}": ["/ES:XCME{=h}", "/MES:XCME{=h}"],
            "/MES:XCME{=h}": ["/ES:XCME{=h}", "/MES:XCME{=h}"],
            "/NQ:XCME{=h}": ["/NQ:XCME{=h}", "/MNQ:XCME{=h}"],
            "/MNQ:XCME{=h}": ["/NQ:XCME{=h}", "/MNQ:XCME{=h}"],
            "/QM:XNYM{=h}": ["/QM:XNYM{=h}", "/MCL:XNYM{=h}"],
            "/MCL:XNYM{=h}": ["/QM:XNYM{=h}", "/MCL:XNYM{=h}"],
        }
        return families.get(symbol, [symbol])
    
    def create_smart_ai_decisions_table(self):
        """Create the Smart_AI_decisions table if it doesn't exist"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            print("Error: Could not connect to database")
            return
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS Smart_AI_decisions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            decision_timestamp TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            predicted_price REAL,
            current_price REAL,
            reasoning TEXT,
            data_sources_used TEXT,
            created_at TEXT NOT NULL
        );
        """
        
        try:
            cursor.execute(create_table_sql)
            conn.commit()
            print("✅ Smart_AI_decisions table is ready")
        except Exception as e:
            print(f"Error creating Smart_AI_decisions table: {e}")
        finally:
            self.db.close_connection(conn)
    
    def extract_json(self, text):
        """Extract JSON from AI response"""
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else text.strip()
    
    def get_historical_data(self, symbol, hours_back=24):
        """Get recent historical data for symbol"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return []
        
        try:
            # Get last 24 hours of data
            query = """
            SELECT open, high, low, close, volume, timestamp
            FROM historical_data_1h
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            cursor.execute(query, (symbol, hours_back))
            rows = cursor.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    'open': row[0],
                    'high': row[1],
                    'low': row[2],
                    'close': row[3],
                    'volume': row[4],
                    'timestamp': row[5]
                })
            return data
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return []
        finally:
            self.db.close_connection(conn)
    
    def check_data_freshness(self, symbol):
        """Check if data sources have recent data (within 1-2 hours)"""
        from datetime import timezone
        current_time = datetime.now(timezone.utc)
        two_hours_ago = current_time - timedelta(hours=2)
        one_hour_ago = current_time - timedelta(hours=1)
        
        freshness_report = {
            'symbol': symbol,
            'historical_data': {'available': False, 'latest_timestamp': None, 'hours_old': None},
            'news_data': {'available': False, 'latest_timestamp': None, 'hours_old': None},
            'news_data_website': {'available': False, 'latest_timestamp': None, 'hours_old': None, 'latest_url': None},
            'lstm_predictions': {'available': False, 'latest_timestamp': None, 'hours_old': None},
            'all_fresh': False,
            'recommendation': 'UPDATE_DATA'
        }
        
        # Check historical data freshness
        conn, cursor = self.db.sql_connect()
        if conn is not None:
            try:
                # Get latest historical data
                query = """
                SELECT timestamp FROM historical_data_1h
                WHERE symbol = ?
                ORDER BY timestamp DESC LIMIT 1
                """
                cursor.execute(query, (symbol,))
                row = cursor.fetchone()
                
                if row:
                    try:
                        # Parse timezone-aware timestamp
                        latest_hist_time = datetime.fromisoformat(row[0])
                        hours_old = (current_time - latest_hist_time).total_seconds() / 3600
                        freshness_report['historical_data'] = {
                            'available': True,
                            'latest_timestamp': row[0],
                            'hours_old': hours_old
                        }
                    except Exception as e:
                        print(f"Error parsing historical timestamp for {symbol}: {e}")
                        freshness_report['historical_data'] = {
                            'available': False,
                            'latest_timestamp': row[0],
                            'hours_old': None
                        }
            except Exception as e:
                print(f"Error checking historical data freshness for {symbol}: {e}")
            
            try:
                # Get latest news data (consider related instruments family)
                related = self.get_related_instruments(symbol)
                placeholders = ",".join(["?"] * len(related))
                like_clause = " OR ".join(["na.instrument LIKE ?" for _ in related])
                query = f"""
                SELECT na.published_at FROM news_articles na
                JOIN sentiment_analysis sa ON na.id = sa.article_id
                WHERE (na.instrument IN ({placeholders}) OR {like_clause})
                ORDER BY na.published_at DESC LIMIT 1
                """
                params = related + [f"%{s}%" for s in related]
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    try:
                        # Parse timezone-aware timestamp
                        latest_news_time = datetime.fromisoformat(row[0])
                        hours_old = (current_time - latest_news_time).total_seconds() / 3600
                        freshness_report['news_data'] = {
                            'available': True,
                            'latest_timestamp': row[0],
                            'hours_old': hours_old
                        }
                    except Exception as e:
                        print(f"Error parsing news timestamp for {symbol}: {e}")
                        freshness_report['news_data'] = {
                            'available': False,
                            'latest_timestamp': row[0],
                            'hours_old': None
                        }
            except Exception as e:
                print(f"Error checking news data freshness for {symbol}: {e}")
            
            # Live website check via prefetched cache (no per-symbol scraping)
            web_info = self.prefetched_news_map.get(symbol)
            if web_info and web_info.get('available'):
                freshness_report['news_data_website'] = dict(web_info)
            
            try:
                # Get latest LSTM predictions
                query = """
                SELECT prediction_timestamp FROM lstm_predictions
                WHERE symbol = ?
                ORDER BY prediction_timestamp DESC LIMIT 1
                """
                cursor.execute(query, (symbol,))
                row = cursor.fetchone()
                
                if row:
                    try:
                        # Parse timezone-aware timestamp
                        latest_lstm_time = datetime.fromisoformat(row[0])
                        hours_old = (current_time - latest_lstm_time).total_seconds() / 3600
                        freshness_report['lstm_predictions'] = {
                            'available': True,
                            'latest_timestamp': row[0],
                            'hours_old': hours_old
                        }
                    except Exception as e:
                        print(f"Error parsing LSTM timestamp for {symbol}: {e}")
                        freshness_report['lstm_predictions'] = {
                            'available': False,
                            'latest_timestamp': row[0],
                            'hours_old': None
                        }
            except Exception as e:
                print(f"Error checking LSTM data freshness for {symbol}: {e}")
            
            self.db.close_connection(conn)
        
        # Determine if all data is fresh (historical within 3 hours, others within 2 hours)
        hist_fresh = (
            freshness_report['historical_data']['available'] and 
            freshness_report['historical_data']['hours_old'] is not None and 
            freshness_report['historical_data']['hours_old'] <= 3
        )
        news_fresh = (
            freshness_report['news_data']['available'] and 
            freshness_report['news_data']['hours_old'] is not None and 
            freshness_report['news_data']['hours_old'] <= 2
        )
        lstm_fresh = (
            freshness_report['lstm_predictions']['available'] and 
            freshness_report['lstm_predictions']['hours_old'] is not None and 
            freshness_report['lstm_predictions']['hours_old'] <= 2
        )
        
        freshness_report['all_fresh'] = hist_fresh and news_fresh and lstm_fresh
        
        if freshness_report['all_fresh']:
            freshness_report['recommendation'] = 'PROCEED'
        else:
            freshness_report['recommendation'] = 'UPDATE_DATA'
        
        return freshness_report
    
    def prompt_user_continuation(self, symbol, freshness_report):
        """Prompt user whether to continue with outdated data"""
        print(f"\n⚠️  DATA FRESHNESS WARNING for {symbol}")
        print("=" * 60)
        
        # Display data freshness status
        print("📊 Data Freshness Status:")
        
        hist_data = freshness_report['historical_data']
        if hist_data['available']:
            print(f"  📈 Historical Data: {hist_data['hours_old']:.1f} hours old")
        else:
            print(f"  📈 Historical Data: NOT AVAILABLE")
        
        news_data = freshness_report['news_data']
        if news_data['available']:
            print(f"  📰 News Data: {news_data['hours_old']:.1f} hours old")
            try:
                if news_data['hours_old'] is not None and news_data['hours_old'] > 2:
                    ts = news_data.get('latest_timestamp') or 'unknown'
                    print(f"  ⚠️  No new news for this symbol since {ts} (≈{news_data['hours_old']:.1f}h old)")
            except Exception:
                pass
        else:
            print(f"  📰 News Data: NOT AVAILABLE")

        # Live website status (URL-based)
        news_web = freshness_report.get('news_data_website', {})
        if news_web.get('available'):
            url_msg = f" | URL: {news_web.get('latest_url')}" if news_web.get('latest_url') else ""
            print(f"  🌐 Website Latest News: {news_web.get('hours_old'):.1f} hours old{url_msg}")
        else:
            print(f"  🌐 Website Latest News: NOT FOUND for related instruments")
        
        lstm_data = freshness_report['lstm_predictions']
        if lstm_data['available']:
            print(f"  🤖 LSTM Predictions: {lstm_data['hours_old']:.1f} hours old")
        else:
            print(f"  🤖 LSTM Predictions: NOT AVAILABLE")
        
        print(f"\n❌ Data is not fresh (older than 2 hours)")
        print("💡 Recommendation: Update data sources before generating signals")
        
        # Handle non-interactive mode
        if not self.interactive_mode:
            print(f"🤖 Non-interactive mode: Skipping {symbol} due to outdated data")
            return False
        
        # Prompt user for continuation
        while True:
            try:
                response = input(f"\n🤔 Do you want to continue with this outdated data for {symbol}? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    print(f"✅ Proceeding with outdated data for {symbol}")
                    return True
                elif response in ['n', 'no']:
                    print(f"❌ Skipping {symbol} due to outdated data")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print(f"\n❌ Skipping {symbol} due to user interruption")
                return False
            except Exception as e:
                print(f"Error in user input: {e}")
                return False
    
    def get_news_sentiment(self, symbol, hours_back=24):
        """Get recent news sentiment for symbol"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return {"avg_sentiment": 0.5, "signal": "NEUTRAL", "article_count": 0}
        
        try:
            # Get news articles and their sentiment; broaden to related instruments family
            related = self.get_related_instruments(symbol)
            placeholders = ",".join(["?"] * len(related))
            like_clause = " OR ".join(["na.instrument LIKE ?" for _ in related])
            query = f"""
            SELECT na.title, na.description, sa.sentiment_score, sa.market_signal
            FROM news_articles na
            JOIN sentiment_analysis sa ON na.id = sa.article_id
            WHERE (na.instrument IN ({placeholders}) OR {like_clause})
              AND datetime(na.published_at) >= datetime('now', '-{hours_back} hours')
            ORDER BY na.published_at DESC
            LIMIT 10
            """
            params = related + [f"%{s}%" for s in related]
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return {"avg_sentiment": 0.5, "signal": "NEUTRAL", "article_count": 0}
            
            sentiments = [row[2] for row in rows if row[2] is not None]
            signals = [row[3] for row in rows if row[3] is not None]
            
            avg_sentiment = mean(sentiments) if sentiments else 0.5
            signal_counter = Counter(signals)
            dominant_signal = signal_counter.most_common(1)[0][0] if signals else "NEUTRAL"
            
            return {
                "avg_sentiment": avg_sentiment,
                "signal": dominant_signal,
                "article_count": len(rows)
            }
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}")
            return {"avg_sentiment": 0.5, "signal": "NEUTRAL", "article_count": 0}
        finally:
            self.db.close_connection(conn)
    
    def get_lstm_predictions(self, symbol, hours_ahead=12):
        """Get LSTM predictions for symbol"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return []
        
        try:
            # Get recent LSTM predictions (up to 12 hours ahead)
            query = """
            SELECT predicted_value, target_timestamp, prediction_timestamp
            FROM lstm_predictions
            WHERE symbol = ?
            AND datetime(target_timestamp) <= datetime('now', '+{} hours')
            AND datetime(target_timestamp) > datetime('now')
            ORDER BY target_timestamp ASC
            LIMIT 12
            """.format(hours_ahead)
            
            cursor.execute(query, (symbol,))
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                predictions.append({
                    'predicted_value': row[0],
                    'target_timestamp': row[1],
                    'prediction_timestamp': row[2]
                })
            return predictions
        except Exception as e:
            print(f"Error fetching LSTM predictions for {symbol}: {e}")
            return []
        finally:
            self.db.close_connection(conn)
    
    def calculate_technical_indicators(self, historical_data):
        """Calculate basic technical indicators"""
        if len(historical_data) < 5:
            return {}
        
        df = pd.DataFrame(historical_data)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Handle NaN values
        df = df.dropna(subset=['close', 'volume'])
        if len(df) < 5:
            return {}
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        # Longer context to stabilize trend signal
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=5).mean()
        
        latest = df.iloc[-1]
        
        return {
            'current_price': latest['close'],
            'sma_5': latest['sma_5'] if not pd.isna(latest['sma_5']) else latest['close'],
            'sma_20': latest['sma_20'] if not pd.isna(latest['sma_20']) else latest['close'],
            'sma_50': latest['sma_50'] if not pd.isna(latest['sma_50']) else latest['close'],
            'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50,
            'volume_ratio': latest['volume'] / latest['volume_sma'] if not pd.isna(latest['volume_sma']) and latest['volume_sma'] > 0 else 1,
            'price_change_24h': ((latest['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100 if len(df) > 0 else 0
        }
    
    def generate_ai_signal(self, symbol, historical_data, news_data, lstm_predictions, technical_indicators):
        """Generate AI signal using GPT-4"""
        
        # Prepare data for AI analysis
        current_price = technical_indicators.get('current_price', 0)
        price_change_24h = technical_indicators.get('price_change_24h', 0)
        rsi = technical_indicators.get('rsi', 50)
        volume_ratio = technical_indicators.get('volume_ratio', 1)
        
        # LSTM predictions summary
        lstm_summary = ""
        if lstm_predictions:
            avg_prediction = mean([p['predicted_value'] for p in lstm_predictions])
            prediction_change = ((avg_prediction - current_price) / current_price) * 100
            lstm_summary = f"LSTM predicts {avg_prediction:.2f} ({prediction_change:+.2f}%)"
        else:
            lstm_summary = "No LSTM predictions available"
        
        # News sentiment summary
        news_summary = f"Sentiment: {news_data['signal']} (Score: {news_data['avg_sentiment']:.2f}, Articles: {news_data['article_count']})"
        
        # Technical analysis summary
        tech_summary = f"RSI: {rsi:.1f}, Volume: {volume_ratio:.2f}x avg, 24h change: {price_change_24h:+.2f}%"
        
        prompt = f"""
You are a financial AI assistant analyzing {symbol} for trading decisions.

📊 **Current Market Data**:
- Current Price: ${current_price:.2f}
- 24h Price Change: {price_change_24h:+.2f}%
- RSI: {rsi:.1f}
- Volume Ratio: {volume_ratio:.2f}x average

📈 **LSTM Predictions**:
{lstm_summary}

📰 **News Sentiment**:
{news_summary}

🎯 **Your Task**:
Analyze all data sources with equal weight and provide a single daily trading signal.
Consider:
1. Historical price trends and technical indicators
2. News sentiment and market signals
3. LSTM model predictions
4. Market volatility and volume patterns

Respond ONLY with JSON in this format:
{{
  "signal": "BUY | HOLD | SELL",
  "confidence_score": 0.85,
  "predicted_price": 1234.56,
  "reasoning": "Brief explanation of decision based on data analysis",
  "data_sources_used": "historical_data,news_sentiment,lstm_predictions"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a financial decision assistant. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            reply = response.choices[0].message.content.strip()
            cleaned = self.extract_json(reply)
            result = json.loads(cleaned)
            
            # Validate and clean the result
            result['confidence_score'] = max(0.0, min(1.0, float(result.get('confidence_score', 0.5))))
            result['predicted_price'] = float(result.get('predicted_price', current_price))
            
            return result
            
        except Exception as e:
            print(f"Error generating AI signal for {symbol}: {e}")
            return {
                "signal": "HOLD",
                "confidence_score": 0.0,
                "predicted_price": current_price,
                "reasoning": f"Error in AI analysis: {str(e)}",
                "data_sources_used": "error"
            }

    def _normalize_news_bias(self, news_signal: str) -> str:
        s = (news_signal or "").strip().lower()
        if s in ("up", "positive", "bullish"):
            return "UP"
        if s in ("down", "negative", "bearish"):
            return "DOWN"
        return "NEUTRAL"

    def _is_trend_up(self, ti: dict) -> bool:
        return (ti.get('sma_5', 0) >= ti.get('sma_20', 0)) or (ti.get('rsi', 50) >= 52)

    def _is_trend_down(self, ti: dict) -> bool:
        return (ti.get('sma_5', 0) <= ti.get('sma_20', 0)) or (ti.get('rsi', 50) <= 48)

    def _apply_alignment_gate(self, signal_data: dict, news_data: dict, lstm_predictions: list, technical_indicators: dict) -> dict:
        """Convert AI signal to 5-level system: STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY
        based on alignment strength and confidence levels.
        """
        current_price = technical_indicators.get('current_price', 0.0) or 0.0
        # LSTM average change
        lstm_change_pct = 0.0
        if lstm_predictions and current_price > 0:
            try:
                avg_pred = mean([p['predicted_value'] for p in lstm_predictions])
                lstm_change_pct = ((avg_pred - current_price) / current_price) * 100.0
            except Exception:
                lstm_change_pct = 0.0

        news_bias = self._normalize_news_bias(news_data.get('signal'))
        trend_up = self._is_trend_up(technical_indicators)
        trend_down = self._is_trend_down(technical_indicators)
        conf = float(signal_data.get('confidence_score', 0.5))
        rsi = technical_indicators.get('rsi', 50)

        # Calculate alignment strength
        alignment_score = 0
        if lstm_change_pct > 0 and news_bias == 'UP' and trend_up:
            alignment_score += 3
        elif lstm_change_pct < 0 and news_bias == 'DOWN' and trend_down:
            alignment_score += 3
        elif (lstm_change_pct > 0 and news_bias == 'UP') or (lstm_change_pct > 0 and trend_up) or (news_bias == 'UP' and trend_up):
            alignment_score += 2
        elif (lstm_change_pct < 0 and news_bias == 'DOWN') or (lstm_change_pct < 0 and trend_down) or (news_bias == 'DOWN' and trend_down):
            alignment_score += 2
        elif lstm_change_pct > 0 or news_bias == 'UP' or trend_up:
            alignment_score += 1
        elif lstm_change_pct < 0 or news_bias == 'DOWN' or trend_down:
            alignment_score += 1

        # Determine 5-level signal based on alignment and confidence
        if alignment_score >= 3 and conf >= 0.8:
            if lstm_change_pct > 0:
                signal_data['signal'] = 'STRONG_BUY'
            else:
                signal_data['signal'] = 'STRONG_SELL'
        elif alignment_score >= 2 and conf >= 0.6:
            if lstm_change_pct > 0:
                signal_data['signal'] = 'BUY'
            else:
                signal_data['signal'] = 'SELL'
        elif alignment_score >= 1 and conf >= 0.4:
            if lstm_change_pct > 0:
                signal_data['signal'] = 'BUY'
            elif lstm_change_pct < 0:
                signal_data['signal'] = 'SELL'
            else:
                signal_data['signal'] = 'HOLD'
        else:
            signal_data['signal'] = 'HOLD'
            signal_data['confidence_score'] = min(conf, 0.5)
            signal_data['reasoning'] = (signal_data.get('reasoning', '') + 
                f" | Gated to HOLD due to weak alignment (score: {alignment_score}, conf: {conf:.2f})").strip()

        return signal_data
    
    def save_decision(self, symbol, signal_data, current_price):
        """Save AI decision to database"""
        conn, cursor = self.db.sql_connect()
        if conn is None:
            return False
        
        try:
            decision_id = str(uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            insert_sql = """
            INSERT INTO Smart_AI_decisions 
            (id, symbol, decision_timestamp, signal, confidence_score, predicted_price, 
             current_price, reasoning, data_sources_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_sql, (
                decision_id,
                symbol,
                timestamp,
                signal_data['signal'],
                signal_data['confidence_score'],
                signal_data['predicted_price'],
                current_price,
                signal_data['reasoning'],
                signal_data['data_sources_used'],
                timestamp
            ))
            
            conn.commit()
            print(f"✅ Saved decision for {symbol}: {signal_data['signal']} (confidence: {signal_data['confidence_score']:.2f})")
            return True
            
        except Exception as e:
            print(f"Error saving decision for {symbol}: {e}")
            return False
        finally:
            self.db.close_connection(conn)
    
    def process_all_symbols(self):
        """Process all 8 symbols and generate daily signals"""
        print("🚀 Starting Refined AI Signal Generation...")
        print(f"Processing {len(self.symbols)} symbols: {', '.join(self.symbols)}")
        
        # 1) Prefetch website news once for all symbols
        self.prefetch_website_news()

        # 2) Build per-symbol freshness reports first
        freshness_by_symbol = {}
        for symbol in self.symbols:
            try:
                freshness_by_symbol[symbol] = self.check_data_freshness(symbol)
            except Exception as e:
                freshness_by_symbol[symbol] = {'error': str(e)}

        # 3) Ask user: process all, subset, or only fresh
        selected_symbols, selection_mode = self.prompt_global_selection(freshness_by_symbol)
        if not selected_symbols:
            print("No symbols selected; exiting.")
            return {}

        results = {}
        skipped_symbols = []
        
        for symbol in self.symbols:
            if symbol not in selected_symbols:
                skipped_symbols.append(symbol)
                results[symbol] = {
                    "status": "SKIPPED",
                    "reason": "Not selected",
                }
                continue

            print(f"\n📊 Processing {symbol}...")
            
            try:
                # Reuse precomputed freshness
                freshness_report = freshness_by_symbol.get(symbol) or self.check_data_freshness(symbol)
                
                # If 'all' mode was chosen, do not prompt per-symbol
                if selection_mode != 'all' and not freshness_report['all_fresh']:
                    print(f"⚠️  Data not fresh for {symbol}")
                    
                    # Prompt user for continuation
                    if not self.prompt_user_continuation(symbol, freshness_report):
                        skipped_symbols.append(symbol)
                        results[symbol] = {
                            "status": "SKIPPED",
                            "reason": "Data not fresh - user chose to skip",
                            "freshness_report": freshness_report
                        }
                        continue
                
                # Get data from all sources
                historical_data = self.get_historical_data(symbol)
                news_data = self.get_news_sentiment(symbol)
                lstm_predictions = self.get_lstm_predictions(symbol)
                
                if not historical_data:
                    print(f"⚠️  No historical data for {symbol}, skipping...")
                    results[symbol] = {
                        "status": "SKIPPED",
                        "reason": "No historical data available"
                    }
                    continue
                
                # Calculate technical indicators
                technical_indicators = self.calculate_technical_indicators(historical_data)
                
                # Generate AI signal
                signal_data = self.generate_ai_signal(
                    symbol, historical_data, news_data, lstm_predictions, technical_indicators
                )
                # Apply alignment gate to increase precision
                signal_data = self._apply_alignment_gate(signal_data, news_data, lstm_predictions, technical_indicators)
                
                # Add freshness info to signal data
                signal_data['data_freshness'] = freshness_report
                
                # Save to database
                current_price = technical_indicators.get('current_price', 0)
                self.save_decision(symbol, signal_data, current_price)
                
                results[symbol] = signal_data
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        # Display summary
        print(f"\n📈 Processing Summary:")
        print(f"  ✅ Successfully processed: {len(results) - len(skipped_symbols)} symbols")
        print(f"  ⏭️  Skipped: {len(skipped_symbols)} symbols")
        
        if skipped_symbols:
            print(f"  📝 Skipped symbols: {', '.join(skipped_symbols)}")
        
        # Save results to JSON file
        output_file = os.path.join(os.path.dirname(__file__), "daily_signals.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Processing complete! Results saved to {output_file}")
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Refined AI Signal Generator')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run in non-interactive mode (skip outdated data without prompting)')
    
    args = parser.parse_args()
    
    processor = RefinedAIProcessor(interactive_mode=not args.non_interactive)
    results = processor.process_all_symbols()
    
    print("\n📈 Daily AI Signals Summary:")
    print("=" * 50)
    for symbol, data in results.items():
        if 'error' not in data and 'status' not in data:
            print(f"{symbol}: {data['signal']} (confidence: {data['confidence_score']:.2f})")
        elif 'status' in data:
            print(f"{symbol}: {data['status']} - {data['reason']}")
        else:
            print(f"{symbol}: ERROR - {data['error']}")

if __name__ == "__main__":
    main()
