import requests
from bs4 import BeautifulSoup as bs
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AzureOpenAI
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import TradeDatabase

import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("api_key"),
    azure_endpoint=os.getenv("azure_endpoint"),
    api_version=os.getenv("api_version")
)

# List of target instruments
target_instruments = [
    "NASDAQ",       # /NQ:XCME
    "S&P 500",      # /ES:XCME
    "Russell 2000", # /RTY:XCME
    "Natural Gas",  # /QG:XNYM
    "Crude Oil",    # /QM:XNYM
    "Bitcoin",      # BTC/USD:CXTALP
    "Ethereum"      # ETH/USD:CXTALP
]

import ast
import re

def detect_instruments_openai(title, description):
    prompt = f"""
    You are a financial analyst. Match this news article to relevant instruments from the list below:

    Instruments:
    {target_instruments}

    News Info:
    Title: {title}
    Description: {description}

    Only return a valid Python list of instrument names. Example:
    ["NASDAQ", "S&P 500"]
    If no instruments match, return ["GENERAL"].
    """

    try:
        response = client.chat.completions.create(
            model="gpt4o",  # ✅ Reference correct deployment
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()

        # Extract Python list
        match = re.search(r"\[(.*?)\]", content)
        if match:
            list_str = "[" + match.group(1) + "]"
            instruments = ast.literal_eval(list_str)
            # Return as comma-separated string for DB storage
            return ", ".join(i for i in instruments if i in target_instruments) or "GENERAL"
        else:
            raise ValueError("No valid list found in model output")

    except Exception as e:
        print(f"OpenAI error: {e}")
        return "GENERAL"


def fetch_description_from_article(url, session, headers):
    """Fetch the meta description from the article page."""
    try:
        if not url.startswith("http"):
            url = "https://finance.yahoo.com" + url
        res = session.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            page_soup = bs(res.content, 'html.parser')
            meta_tag = page_soup.find('meta', attrs={'name': 'description'})
            if meta_tag and meta_tag.get('content'):
                return meta_tag['content'].strip()
    except Exception as e:
        print(f"Failed to fetch description for {url}: {e}")
    return "No description available"


def get_data():
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    session = requests.Session()
    response = session.get(url, headers=headers)

    news_list = []
    if response.status_code == 200:
        soup = bs(response.content, 'html.parser')
        articles = soup.find_all('section', class_='container')

        for article in articles:
            title_tag = article.find('h3', class_='clamp')
            title = title_tag.text.strip() if title_tag else "No title available"
            if title == "No title available":
                continue

            image_tag = article.find('img')
            image = image_tag['src'] if image_tag else "No image available"

            source_url_tag = article.find('a')
            if source_url_tag and source_url_tag.get('href'):
                href = source_url_tag['href']
                source_url = href if href.startswith('http') else f"https://finance.yahoo.com{href}"
            else:
                source_url = "No source URL available"

            # ✅ Fetch description from individual article page
            description = fetch_description_from_article(source_url, session, headers)

            footer_div = article.find('div', class_='footer')
            if footer_div:
                footer_tag = footer_div.find('div', class_='publishing')
                publisher = footer_tag.text.strip() if footer_tag else "No footer available"
            else:
                publisher = "No footer available"

            # Detect related instruments using OpenAI
            instrument = detect_instruments_openai(title, description)

            news_list.append({
                'id': str(uuid.uuid4()),
                'title': title,
                'description': description,
                'published_at': datetime.now(timezone.utc).isoformat(),
                'url': source_url,
                'source': publisher,
                'instrument': instrument
            })

    return news_list


def store_news_to_sqlite(news_list):
    conn, cursor = TradeDatabase.sql_connect()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            published_at TEXT,
            url TEXT,
            source TEXT,
            instrument TEXT
        )
    """)
    for news in news_list:
        cursor.execute("""
            INSERT OR IGNORE INTO news_articles 
            (id, title, description, published_at, url, source, instrument)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            news['id'], news['title'], news['description'],
            news['published_at'], news['url'], news['source'], news['instrument']
        ))
    conn.commit()
    TradeDatabase.close_connection(conn)

if __name__ == "__main__":
    news_data = get_data()
    print(f"Fetched {len(news_data)} articles.")
    store_news_to_sqlite(news_data)
