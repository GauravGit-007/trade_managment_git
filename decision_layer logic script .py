#this script is brought up from the prediction model folder to here,for the decision layer logic
# Decision Layer Logic Script
import os
import json
import re
from collections import defaultdict, Counter
from statistics import mean
from datetime import datetime, timedelta
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding='utf-8')

# DB setup
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.market_database import toolsDB

# Load API keys
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("api_key"),
    azure_endpoint=os.getenv("azure_endpoint"),
    api_version=os.getenv("api_version")
)


def extract_json(text):
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else text.strip()


def load_json(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path) as f:
        return json.load(f)


# Load data files
price_data = load_json("future_price_forecasts.json")
macro_data = load_json("llm_event_sentiment.json")
news_data = load_json("news_model_output.json")

# Map macro and news data for quick lookup
macro_map = defaultdict(list)
for entry in macro_data:
    macro_map[entry["instrument"]].append(entry)

news_map = defaultdict(list)
for entry in news_data:
    instruments = (
        [s.strip() for s in entry["instrument"].split(",")]
        if isinstance(entry["instrument"], str) else [entry["instrument"]]
    )
    for symbol in instruments:
        news_map[symbol].append(entry)


def summarize_historical(symbol, upcoming_indicators):
    impacts = []
    for indicator in upcoming_indicators:
        for entry in macro_map.get(symbol, []):
            if entry["indicator"] == indicator and "impact_label" in entry:
                impacts.append(entry["impact_label"])
    if not impacts:
        return "insufficient data"
    top_impact = Counter(impacts).most_common(1)[0]
    return f"{top_impact[0]} ({top_impact[1]} occurrences)"


def get_recent_news_entries(symbol, as_of_date):
    entries = []
    for entry in news_map.get(symbol, []):
        try:
            pubdate = datetime.strptime(entry.get("published_at"), "%Y-%m-%d").date()
            if pubdate <= as_of_date:
                entries.append(entry)
        except:
            continue
    return entries


# 🆕 Fetch predicted vs actual values for today
def fetch_actual_vs_predicted(symbol, target_date):
    """
    Fetch predicted and actual values for `symbol` on `target_date`.
    """
    try:
        conn, cursor = toolsDB.sql_connect()
        query = """
            SELECT predicted_value, actual_value
            FROM model_predictions
            WHERE symbol = ? AND prediction_date = ? AND model_name = 'LSTM'
            LIMIT 1
        """
        cursor.execute(query, (symbol, target_date))
        row = cursor.fetchone()
        toolsDB.close_connection(conn)

        if row and row[0] is not None and row[1] is not None:
            predicted = row[0]
            actual = row[1]
            difference = actual - predicted
            pct_diff = (difference / predicted) * 100 if predicted != 0 else 0
            return predicted, actual, difference, pct_diff
        else:
            return None, None, None, None
    except Exception as e:
        print(f"[ERROR] Fetching actual vs predicted for {symbol} on {target_date}: {e}")
        return None, None, None, None


# === MAIN DECISION LAYER ===
decisions = {}
today = datetime.utcnow().date()

for symbol, forecast_obj in price_data.items():
    current = forecast_obj["current"]
    forecasts = forecast_obj["forecast"]
    current_atr = forecast_obj.get("ATR_14", None)  # ✅ ATR value
    candles = forecast_obj.get("recent_candles", [])

    # ✅ Trend direction
    recent_closes = [
        float(candle["close"])
        for candle in candles[-5:]
        if candle.get("close") is not None
    ]
    if len(recent_closes) >= 2:
        trend_direction = "upward" if recent_closes[-1] > recent_closes[0] else "downward"
    else:
        trend_direction = "unknown"

    symbol_decisions = []

    # 🆕 Fetch actual vs predicted for today
    predicted_value, actual_value, difference, pct_diff = fetch_actual_vs_predicted(symbol, today)

    # 🆕 Safely format values
    predicted_value_str = f"{predicted_value:.2f}" if predicted_value is not None else "N/A"
    actual_value_str = f"{actual_value:.2f}" if actual_value is not None else "N/A"
    difference_str = (
        f"{difference:+.2f} ({pct_diff:+.2f}%)" if difference is not None else "N/A"
    )

    if predicted_value is not None and actual_value is not None:
        bias_info = (
            f"LSTM predicted: {predicted_value_str}, "
            f"Actual: {actual_value_str}, "
            f"Difference: {difference_str}"
        )
    else:
        bias_info = "No actual value available for today to compute prediction error."

    for i, forecast in enumerate(forecasts):
        forecast_date = today + timedelta(days=i + 1)

        # Macroeconomic data
        macro_entries = [e for e in macro_map.get(symbol, []) if str(e.get("date")) == str(forecast_date)]
        macro_indicators = [e["indicator"] for e in macro_entries]
        macro_impacts = [e["impact_label"] for e in macro_entries]
        macro_sentiments = [e["sentiment_label"] for e in macro_entries]
        macro_summary = Counter(macro_impacts).most_common(1)[0][0] if macro_impacts else "Unknown"
        macro_sent_summary = Counter(macro_sentiments).most_common(1)[0][0] if macro_sentiments else "Unknown"
        historical_summary = summarize_historical(symbol, macro_indicators)

        # News data
        news_entries = get_recent_news_entries(symbol, today)
        avg_sentiment = mean([e["sentiment_score"] for e in news_entries]) if news_entries else 0.5
        news_signals = [e["market_signal"] for e in news_entries if e.get("market_signal")]
        news_signal = Counter(news_signals).most_common(1)[0][0] if news_signals else "Unknown"

        # Prompt
        prompt = f"""
You are a financial AI assistant tasked with improving price forecasts and trade recommendations. Use the following data for {symbol} on {forecast_date}: 

📈 **Forecast Data**:
- Current price: {current:.2f}
- LSTM predicted price (today): {predicted_value_str}
- Actual price (today): {actual_value_str}
- Difference: {difference_str}
- LSTM forecasted price (future): {forecast:.2f}
- Forecasted % change: {(forecast - current)/current*100:.2f}%
- Trend direction: {trend_direction}

📉 **Prediction Accuracy Context**:
The LSTM predicted value today differed from the actual price as shown above. Adjust the future forecast by considering this prediction error (use it as a bias correction).

📊 **Macroeconomic Events**:
Indicators: {macro_indicators}
Likely impact: {macro_summary}
Sentiment: {macro_sent_summary}

📰 **News Sentiment**:
Signal: {news_signal}
Avg. sentiment score: {avg_sentiment:.2f}

🎯 **Your Task**:
1. Adjust the future forecast by analyzing how much the previous LSTM prediction differed from the actual price today. Apply this as a bias correction for future forecasts.
2. Incorporate macroeconomic indicators, ATR volatility, and news sentiment in your adjustment.
3. Recommend BUY, HOLD, or SELL based on the adjusted forecast.
4. Justify your decision clearly but do NOT mention bias correction or that adjustments were made.

Respond ONLY with JSON in this format:
{{
  "date": "{forecast_date}",
  "forecast": float,
  "recommendation": "BUY | HOLD | SELL",
  "reason": Short concise reason, e.g., 'Forecasted price rise of +5.27% supports a BUY recommendation.',
  "confidence_score": float (0.0 to 1.0)
}}
"""


        try:
            response = client.chat.completions.create(
                model="gpt4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a financial decision assistant. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            reply = response.choices[0].message.content.strip()
            cleaned = extract_json(reply)
            result = json.loads(cleaned)
            result["confidence_score"] = round(float(result.get("confidence_score", 0.5)), 3)

            # ✅ Save LLM-adjusted forecast 
            toolsDB.log_llm_prediction(
                symbol=symbol,
                prediction_date=str(forecast_date),
                model_name="LSTM",
                predicted=forecast,
                adjusted=result["forecast"],
                recommendation=result["recommendation"],
                reason=result["reason"],
                confidence=result["confidence_score"]
            )

        except Exception as e:
            print(f"Error with {symbol} on {forecast_date}: {e}")
            result = {
                "date": str(forecast_date),
                "forecast": forecast,
                "recommendation": "ERROR",
                "reason": str(e),
                "confidence_score": 0.0
            }

        symbol_decisions.append(result)

    decisions[symbol] = symbol_decisions

# Save all decisions to JSON
output_path = os.path.join("trade_recommendations_by_day.json")
with open(output_path, "w") as f:
    json.dump(decisions, f, indent=2)

print(f"✅ Saved trade recommendations to '{output_path}'")
