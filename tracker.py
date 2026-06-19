import feedparser
import json
import os
import requests
from groq import Groq
from datetime import datetime

# --- Config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# --- Finance News RSS Sources ---
RSS_SOURCES = [
    {"name": "Reuters Finance", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "CNBC Finance", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html"},
    {"name": "Bloomberg Markets", "url": "https://feeds.bloomberg.com/markets/news.rss"},
    {"name": "WSJ Markets", "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"},
    {"name": "Financial Times", "url": "https://www.ft.com/rss/home"},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/feed.xml"},
]

# --- Important Finance Keywords (Layer 1 Filter) ---
IMPORTANT_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "inflation", "recession",
    "gdp", "unemployment", "cpi", "fomc", "rate hike", "rate cut",
    "banking", "bank collapse", "earnings", "ipo", "market crash",
    "bull market", "bear market", "hedge fund", "treasury", "bonds",
    "stock market", "s&p", "nasdaq", "dow jones", "wall street",
    "debt ceiling", "fiscal", "monetary policy", "quantitative",
    "powell", "yellen", "blackrock", "jpmorgan", "goldman sachs"
]

def fetch_news():
    """Fetch news from all RSS sources"""
    all_news = []
    for source in RSS_SOURCES:
        try:
            feed = feedparser.parse(source["url"])
            for entry in feed.entries[:15]:
                all_news.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                    "source": source["name"],
                    "link": entry.get("link", ""),
                    "date": entry.get("published", "")
                })
            print(f"✓ {source['name']}: {len(feed.entries[:15])} articles")
        except Exception as e:
            print(f"✗ {source['name']}: {e}")
    return all_news

def keyword_filter(news_list):
    """Layer 1: Fast keyword-based filter"""
    filtered = []
    for item in news_list:
        text = (item["title"] + " " + item["summary"]).lower()
        if any(kw in text for kw in IMPORTANT_KEYWORDS):
            filtered.append(item)
    print(f"Layer 1 filter: {len(news_list)} → {len(filtered)} articles")
    return filtered

def groq_filter_and_summarize(news_list):
    """Layer 2: Groq scores importance and summarizes top news"""
    if not news_list:
        return []

    # Prepare headlines for Groq
    headlines = "\n".join([
        f"{i+1}. [{item['source']}] {item['title']}"
        for i, item in enumerate(news_list[:40])
    ])

    prompt = f"""You are a senior financial analyst. Today's date: {datetime.now().strftime('%Y-%m-%d')}.

Below are finance news headlines. Your job:
1. Score each from 1-10 (market impact importance)
2. Keep only score 7 or above
3. Group them into categories
4. Write a brief insight for each

Headlines:
{headlines}

Return ONLY this JSON format, no extra text:
{{
  "date": "{datetime.now().strftime('%Y-%m-%d')}",
  "top_stories": [
    {{
      "rank": 1,
      "title": "headline",
      "source": "source name",
      "category": "Fed Policy / Inflation / Banking / Markets / Earnings / Global",
      "importance_score": 9,
      "insight": "Why this matters in 1-2 sentences"
    }}
  ],
  "daily_summary": "Overall market narrative for today in 3-4 sentences",
  "key_themes": ["theme1", "theme2", "theme3"],
  "market_sentiment": "bullish/bearish/neutral/mixed"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        raw = response.choices[0].message.content.strip()
        # Clean JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def main():
    print(f"Starting tracker: {datetime.now().isoformat()}")

    # Step 1: Fetch
    news = fetch_news()
    print(f"Total fetched: {len(news)}")

    # Step 2: Layer 1 filter
    filtered = keyword_filter(news)

    # Step 3: Layer 2 Groq filter + summarize
    result = groq_filter_and_summarize(filtered)

    if not result:
        print("Groq failed. Saving raw data.")
        result = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "error": "Groq processing failed",
            "raw_count": len(filtered)
        }

    # Add metadata
    result["updated"] = datetime.now().isoformat()
    result["sources_checked"] = len(RSS_SOURCES)
    result["articles_processed"] = len(filtered)

    # Save
    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to docs/data.json")
    if "top_stories" in result:
        print(f"Top stories: {len(result['top_stories'])}")

if __name__ == "__main__":
    main()
