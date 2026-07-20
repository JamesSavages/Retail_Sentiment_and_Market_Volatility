"""
ONE-REQUEST test: does Alpha Vantage NEWS_SENTIMENT accept MULTIPLE tickers in a
single request and still return per-ticker sentiment scores for each?

If yes, the collection quota maths changes dramatically: instead of
  24 requests per ticker  (8 tickers = 192 requests = ~8 days)
we could fetch several tickers per request, cutting the collection time.

IMPORTANT CAVEAT this test checks for: when you pass tickers=A,B,C, Alpha Vantage
returns articles relevant to ANY of them, capped by `limit`. So an article about
AAPL may not mention NVDA at all. We must verify each ticker actually receives a
usable number of its own articles, not just that the request succeeds.

Run:
    uv run src/test_multiticker.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)
AV_URL = "https://www.alphavantage.co/query"


def main() -> None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or key.lower() == "your_key_here":
        print("ERROR: set a real ALPHAVANTAGE_API_KEY in .env")
        return

    tickers = "AAPL,NVDA,TSLA"
    print(f"Testing ONE request for multiple tickers: {tickers}")
    print("Window: January 2023 (same as a normal monthly pull)\n")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "time_from": "20230101T0000",
        "time_to": "20230131T2359",
        "limit": 1000,
        "sort": "EARLIEST",
        "apikey": key,
    }
    try:
        r = requests.get(AV_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        return

    if "Note" in data or "Information" in data:
        print(f"QUOTA/LIMIT: {data.get('Note') or data.get('Information')}")
        return

    feed = data.get("feed", [])
    print(f"Articles returned in ONE request: {len(feed)}")
    if not feed:
        print("No articles returned.")
        return

    # How many articles carry a sentiment score for EACH requested ticker?
    wanted = tickers.split(",")
    per_ticker = Counter()
    for item in feed:
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") in wanted and ts.get("ticker_sentiment_score") is not None:
                per_ticker[ts["ticker"]] += 1

    print("\nArticles WITH a per-ticker sentiment score, by ticker:")
    for tkr in wanted:
        print(f"  {tkr:6s} {per_ticker.get(tkr, 0):4d}")

    dates = sorted(a.get("time_published", "") for a in feed)
    print(f"\nDate range covered: {dates[0]} .. {dates[-1]}")

    print("\n--- INTERPRETATION ---")
    if len(feed) >= 950:
        print("WARNING: hit the ~1000-article limit. With multiple tickers the feed is")
        print("shared, so each ticker may be TRUNCATED (esp. later in the month).")
        print("Compare the date range above against the full month: if it stops early,")
        print("multi-ticker requests are LOSING data and per-ticker pulls are safer.")
    else:
        print("Feed is under the limit, so no truncation: each ticker's articles are")
        print("fully covered. Multi-ticker requests look SAFE for this window.")
    if min(per_ticker.get(t, 0) for t in wanted) == 0:
        print("NOTE: at least one ticker got 0 scored articles - multi-ticker is unreliable.")


if __name__ == "__main__":
    main()