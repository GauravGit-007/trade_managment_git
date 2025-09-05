from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import pandas as pd
import os
import sys
import json
from datetime import datetime
import uuid

# Import toolsDB from your project (adjust import if your path is different)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase as toolsDB

# Load FinBERT tokenizer and model
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load news articles from SQLite using toolsDB
def get_news_from_db():
    conn, cursor = toolsDB.sql_connect()
    query = "SELECT id, title, description, published_at, url, source, instrument FROM news_articles"
    df = pd.read_sql_query(query, conn)
    toolsDB.close_connection(conn)
    return df

# Map model labels to market signals
def map_sentiment_to_class(label):
    if label == "positive":
        return "Up"
    elif label == "negative":
        return "Down"
    else:
        return "Flat"

# Save results into sentiment_analysis table
def save_sentiments_to_db(news_df):
    conn, cursor = toolsDB.sql_connect()
    now = datetime.utcnow().isoformat()

    for _, row in news_df.iterrows():
        sentiment_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT OR REPLACE INTO sentiment_analysis
            (id, article_id, sentiment_label, sentiment_score, market_signal, analysis_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sentiment_id,
                row["id"],
                row["sentiment_label"],
                row["sentiment_score"],
                row["market_signal"],
                now,
            )
        )

    conn.commit()
    toolsDB.close_connection(conn)
    print(f"✅ Inserted {len(news_df)} sentiment records into sentiment_analysis")

def run_pipeline():
    news_df = get_news_from_db()

    # Prepare text for sentiment analysis
    news_df["text"] = news_df["title"].fillna("") + ". " + news_df["description"].fillna("")
    news_df["published_at"] = pd.to_datetime(news_df["published_at"]).dt.date

    # Run model
    subset = news_df["text"].tolist()
    results = sentiment_pipeline(subset, truncation=True)

    # Add results
    news_df["sentiment_label"] = [res["label"].lower() for res in results]
    news_df["sentiment_score"] = [float(res["score"]) for res in results]
    news_df["market_signal"] = news_df["sentiment_label"].map(map_sentiment_to_class)

    # Save to DB
    save_sentiments_to_db(news_df)

    # Also save JSON for LLM
    llm_ready = [
        {
            "published_at": str(row["published_at"]),
            "text": row["text"],
            "instrument": row.get("instrument", ""),
            "source": row.get("source", ""),
            "url": row.get("url", ""),
            "market_signal": row["market_signal"],
            "sentiment_label": row["sentiment_label"],
            "sentiment_score": row["sentiment_score"],
        }
        for _, row in news_df.iterrows()
    ]

    output_path = os.path.join("llm_decision_layer", "news_model_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(llm_ready, f, indent=2, default=str)

    print(f"📄 News sentiment output saved to: {output_path}")
    return llm_ready, output_path


if __name__ == "__main__":
    run_pipeline()
