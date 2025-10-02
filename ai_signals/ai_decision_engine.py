# AI Decision Engine for Trade Management
# This module implements AI-based trading decisions using Azure OpenAI
# Based on the decision layer logic script but enhanced for better accuracy

import os
import json
import re
import sys
from collections import defaultdict, Counter
from statistics import mean
from datetime import datetime, timedelta
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add parent directory to path for database access
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase

# Configure encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

class AIDecisionEngine:
    def __init__(self):
        """Initialize the AI Decision Engine with Azure OpenAI credentials"""
        self.client = AzureOpenAI(
            api_key="71b66107a84e489ea700ef4188d29947",
            azure_endpoint="https://vastai-openai-swedencentral.openai.azure.com/",
            api_version="2024-02-15-preview"
        )
        
        # Trading symbols we work with
        self.symbols = [
            "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}", 
            "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}", 
            "BTC/USD:CXTALP{=h}", "ETH/USD:CXTALP{=h}"
        ]
        
        # Action mapping for consistency with existing system
        self.action_mapping = {
            "STRONG_SELL": 0,
            "SELL": 1, 
            "HOLD": 2,
            "BUY": 3,
            "STRONG_BUY": 4
        }
        
        # Initialize database connection
        self.db = TradeDatabase()
        
    def extract_json(self, text):
        """Extract JSON from AI response"""
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else text.strip()
    
    def get_current_price_data(self, symbol):
        """Get current price data from database"""
        try:
            conn, cursor = self.db.sql_connect()
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                AND open IS NOT NULL 
                AND high IS NOT NULL 
                AND low IS NOT NULL 
                AND close IS NOT NULL 
                AND volume IS NOT NULL
                AND open != 'NaN' 
                AND high != 'NaN' 
                AND low != 'NaN' 
                AND close != 'NaN' 
                AND volume != 'NaN'
                AND open != '' 
                AND high != '' 
                AND low != '' 
                AND close != '' 
                AND volume != ''
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            cursor.execute(query, (symbol,))
            row = cursor.fetchone()
            self.db.close_connection(conn)
            
            if row:
                # Additional validation for numeric values
                try:
                    return {
                        'open': float(row[0]) if row[0] is not None else None,
                        'high': float(row[1]) if row[1] is not None else None,
                        'low': float(row[2]) if row[2] is not None else None,
                        'close': float(row[3]) if row[3] is not None else None,
                        'volume': int(row[4]) if row[4] is not None else None,
                        'timestamp': row[5]
                    }
                except (ValueError, TypeError) as e:
                    print(f"Invalid numeric data for {symbol}: {e}")
                    return None
            return None
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_historical_candles(self, symbol, lookback_hours=24):
        """Get historical candle data for analysis"""
        try:
            conn, cursor = self.db.sql_connect()
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                AND open IS NOT NULL 
                AND high IS NOT NULL 
                AND low IS NOT NULL 
                AND close IS NOT NULL 
                AND volume IS NOT NULL
                AND open != 'NaN' 
                AND high != 'NaN' 
                AND low != 'NaN' 
                AND close != 'NaN' 
                AND volume != 'NaN'
                AND open != '' 
                AND high != '' 
                AND low != '' 
                AND close != '' 
                AND volume != ''
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            cursor.execute(query, (symbol, lookback_hours))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            candles = []
            for row in rows:
                try:
                    # Validate and convert numeric values
                    candle = {
                        'open': float(row[0]) if row[0] is not None else None,
                        'high': float(row[1]) if row[1] is not None else None,
                        'low': float(row[2]) if row[2] is not None else None,
                        'close': float(row[3]) if row[3] is not None else None,
                        'volume': int(row[4]) if row[4] is not None else None,
                        'timestamp': row[5]
                    }
                    
                    # Only add if all numeric values are valid
                    if all(candle[key] is not None for key in ['open', 'high', 'low', 'close', 'volume']):
                        candles.append(candle)
                        
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid candle data for {symbol}: {e}")
                    continue
                    
            return candles
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def get_news_sentiment(self, symbol, hours_back=24):
        """Get recent news sentiment for symbol"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Get news articles for the symbol
            query = """
                SELECT na.title, na.description, na.published_at, sa.sentiment_label, 
                       sa.sentiment_score, sa.market_signal
                FROM news_articles na
                LEFT JOIN sentiment_analysis sa ON na.id = sa.article_id
                WHERE na.instrument LIKE ? 
                AND datetime(na.published_at) >= datetime('now', '-{} hours')
                ORDER BY na.published_at DESC
            """.format(hours_back)
            
            cursor.execute(query, (f"%{symbol.split(':')[0].replace('/', '')}%",))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            if not rows:
                return {
                    'avg_sentiment': 0.5,
                    'market_signal': 'NEUTRAL',
                    'article_count': 0,
                    'recent_news': []
                }
            
            sentiments = []
            market_signals = []
            recent_news = []
            
            for row in rows:
                if row[3]:  # sentiment_label exists
                    sentiments.append(row[4])  # sentiment_score
                    market_signals.append(row[5])  # market_signal
                
                recent_news.append({
                    'title': row[0],
                    'description': row[1],
                    'published_at': row[2],
                    'sentiment': row[4] if row[4] else 0.5,
                    'signal': row[5] if row[5] else 'NEUTRAL'
                })
            
            avg_sentiment = mean(sentiments) if sentiments else 0.5
            dominant_signal = Counter(market_signals).most_common(1)[0][0] if market_signals else 'NEUTRAL'
            
            return {
                'avg_sentiment': avg_sentiment,
                'market_signal': dominant_signal,
                'article_count': len(rows),
                'recent_news': recent_news[:5]  # Last 5 articles
            }
            
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}")
            return {
                'avg_sentiment': 0.5,
                'market_signal': 'NEUTRAL', 
                'article_count': 0,
                'recent_news': []
            }
    
    def calculate_technical_indicators(self, candles):
        """Calculate basic technical indicators"""
        if len(candles) < 14:
            return {}
        
        df = pd.DataFrame(candles)
        
        # Convert to numeric, handling NaN values and invalid strings
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Additional validation - remove any remaining invalid data
        df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['volume'] >= 0)]
        
        if len(df) < 14:
            return {}
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Volume trend
        df['volume_sma'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(5)
        
        latest = df.iloc[-1]
        
        return {
            'current_price': float(latest['close']),
            'sma_5': float(latest['sma_5']) if not pd.isna(latest['sma_5']) else None,
            'sma_20': float(latest['sma_20']) if not pd.isna(latest['sma_20']) else None,
            'rsi': float(latest['rsi']) if not pd.isna(latest['rsi']) else None,
            'atr': float(latest['atr']) if not pd.isna(latest['atr']) else None,
            'volume_ratio': float(latest['volume_ratio']) if not pd.isna(latest['volume_ratio']) else None,
            'price_momentum_5': float(latest['momentum_5']) if not pd.isna(latest['momentum_5']) else None,
            'trend_direction': 'UP' if latest['sma_5'] > latest['sma_20'] else 'DOWN' if latest['sma_5'] < latest['sma_20'] else 'SIDEWAYS'
        }
    
    def get_market_context(self, symbol):
        """Get comprehensive market context for decision making"""
        # Get current price data
        current_data = self.get_current_price_data(symbol)
        if not current_data or current_data['close'] is None:
            print(f"❌ No valid current price data for {symbol}")
            return None
        
        # Get historical candles for technical analysis
        candles = self.get_historical_candles(symbol, 24)
        if not candles or len(candles) < 5:
            print(f"❌ Insufficient historical data for {symbol} (got {len(candles)} candles)")
            return None
        
        # Calculate technical indicators
        technicals = self.calculate_technical_indicators(candles)
        if not technicals:
            print(f"❌ Could not calculate technical indicators for {symbol}")
            return None
        
        # Get news sentiment
        news_data = self.get_news_sentiment(symbol, 24)
        
        # Get current position
        current_position = self.db.get_current_position(symbol.split('{')[0])
        
        return {
            'symbol': symbol,
            'current_price': current_data['close'],
            'current_position': current_position,
            'technical_indicators': technicals,
            'news_sentiment': news_data,
            'price_data': {
                'open': current_data['open'],
                'high': current_data['high'],
                'low': current_data['low'],
                'close': current_data['close'],
                'volume': current_data['volume']
            },
            'timestamp': current_data['timestamp']
        }
    
    def generate_ai_signal(self, market_context):
        """Generate AI trading signal based on comprehensive market analysis"""
        symbol = market_context['symbol']
        current_price = market_context['current_price']
        technicals = market_context['technical_indicators']
        news = market_context['news_sentiment']
        current_position = market_context['current_position']
        
        # Build comprehensive prompt for AI
        prompt = f"""
You are an expert quantitative trader with access to real-time market data. Analyze the following information for {symbol} and provide a trading recommendation.

📊 **CURRENT MARKET DATA**:
- Current Price: ${current_price:.2f}
- Current Position: {current_position} units
- Timestamp: {market_context['timestamp']}

📈 **TECHNICAL ANALYSIS**:
- 5-period SMA: ${technicals.get('sma_5', 'N/A')}
- 20-period SMA: ${technicals.get('sma_20', 'N/A')}
- RSI (14): {technicals.get('rsi', 'N/A')}
- ATR (14): ${technicals.get('atr', 'N/A')}
- Volume Ratio: {technicals.get('volume_ratio', 'N/A')}
- 5-period Momentum: {technicals.get('price_momentum_5', 'N/A')}
- Trend Direction: {technicals.get('trend_direction', 'UNKNOWN')}

📰 **NEWS SENTIMENT**:
- Average Sentiment Score: {news['avg_sentiment']:.3f} (0=very negative, 1=very positive)
- Market Signal: {news['market_signal']}
- Recent Articles: {news['article_count']}
- Latest News Headlines: {[n['title'][:50] + '...' for n in news['recent_news'][:3]]}

🎯 **YOUR TASK**:
Analyze all available data and provide a trading recommendation. Consider:
1. Technical indicators alignment
2. News sentiment impact
3. Current position and risk management
4. Market volatility and trends
5. Risk-reward ratio

Respond ONLY with valid JSON in this exact format:
{{
    "action": "STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of your decision based on technical and fundamental analysis",
    "price_target": 4250.50,
    "stop_loss": 4200.00,
    "risk_reward_ratio": 2.5,
    "key_factors": ["RSI oversold", "Positive news sentiment", "Volume confirmation"],
    "time_horizon": "1-4 hours"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt4o",
                temperature=0.2,  # Lower temperature for more consistent decisions
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trader. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            reply = response.choices[0].message.content.strip()
            cleaned = self.extract_json(reply)
            result = json.loads(cleaned)
            
            # Validate and enhance the result
            result['symbol'] = symbol
            result['timestamp'] = datetime.utcnow().isoformat()
            result['current_price'] = current_price
            result['action_code'] = self.action_mapping.get(result['action'], 2)  # Default to HOLD
            
            return result
            
        except Exception as e:
            print(f"Error generating AI signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'action_code': 2,
                'confidence': 0.0,
                'reasoning': f"Error in AI analysis: {str(e)}",
                'timestamp': datetime.utcnow().isoformat(),
                'current_price': current_price
            }
    
    def log_decision(self, decision):
        """Log the AI decision to database"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Create AI decisions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    decision_timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    action_code INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    price_target REAL,
                    stop_loss REAL,
                    risk_reward_ratio REAL,
                    key_factors TEXT,
                    time_horizon TEXT,
                    current_price REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert decision
            decision_id = f"ai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{decision['symbol'].replace(':', '_').replace('/', '_')}"
            
            cursor.execute("""
                INSERT INTO ai_decisions 
                (id, symbol, decision_timestamp, action, action_code, confidence, reasoning, 
                 price_target, stop_loss, risk_reward_ratio, key_factors, time_horizon, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id,
                decision['symbol'],
                decision['timestamp'],
                decision['action'],
                decision['action_code'],
                decision['confidence'],
                decision['reasoning'],
                decision.get('price_target'),
                decision.get('stop_loss'),
                decision.get('risk_reward_ratio'),
                json.dumps(decision.get('key_factors', [])),
                decision.get('time_horizon'),
                decision['current_price']
            ))
            
            conn.commit()
            self.db.close_connection(conn)
            
            print(f"✅ Logged AI decision for {decision['symbol']}: {decision['action']} (confidence: {decision['confidence']:.2f})")
            
        except Exception as e:
            print(f"Error logging AI decision: {e}")
    
    def generate_signals_for_all_symbols(self):
        """Generate AI signals for all trading symbols"""
        all_decisions = {}
        
        print("🤖 Starting AI signal generation for all symbols...")
        
        for symbol in self.symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get market context
            market_context = self.get_market_context(symbol)
            if not market_context:
                print(f"❌ No data available for {symbol}")
                continue
            
            # Generate AI signal
            decision = self.generate_ai_signal(market_context)
            
            # Log decision
            self.log_decision(decision)
            
            # Store for output
            all_decisions[symbol] = decision
            
            print(f"✅ Generated signal for {symbol}: {decision['action']} (confidence: {decision['confidence']:.2f})")
        
        # Save all decisions to JSON file
        output_path = os.path.join(os.path.dirname(__file__), "ai_signals_output.json")
        with open(output_path, "w") as f:
            json.dump(all_decisions, f, indent=2)
        
        print(f"\n✅ All AI signals saved to {output_path}")
        return all_decisions

if __name__ == "__main__":
    # Initialize and run the AI decision engine
    engine = AIDecisionEngine()
    decisions = engine.generate_signals_for_all_symbols()
    
    # Print summary
    print("\n📋 AI SIGNALS SUMMARY:")
    print("=" * 50)
    for symbol, decision in decisions.items():
        print(f"{symbol}: {decision['action']} (confidence: {decision['confidence']:.2f})")
