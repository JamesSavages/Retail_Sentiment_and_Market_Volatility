"""
One-shot smoke test for the two network sources, BEFORE any full pull.

It makes exactly ONE request to each source so you don't burn quota:
  * Alpha Vantage NEWS_SENTIMENT  -> confirms free-tier access AND whether
    historical (2023) news is actually returned.
  * StockTwits stream             -> confirms the public endpoint is reachable
    and that messages carry self-tagged Bullish/Bearish labels.

Run:
    uv run src/test_sources.py
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)


def test_alpha_vantage_news() -> None:
    print("\n=== Alpha Vantage NEWS_SENTIMENT (1 request) ===")
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("FAIL: ALPHAVANTAGE_API_KEY not set to a real key in .env")
        return
    params = {
        "function": "NEWS_SENTIMENT", "tickers": "AAPL",
        "time_from": "20230103T0000", "time_to": "20230131T2359",
        "limit": 50, "sort": "EARLIEST", "apikey": key,
    }
    try:
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: request error: {exc}")
        return

    if "Information" in data or "Note" in data:
        print(f"  GATED/LIMITED: {data.get('Information') or data.get('Note')}")
        return
    feed = data.get("feed", [])
    if not feed:
        print("  WARNING: no articles returned for the 2023 window (empty feed).")
        return
    dates = sorted(a.get("time_published", "") for a in feed)
    print(f"  PASS: {len(feed)} articles returned")
    print(f"  date range in response: {dates[0]}  ..  {dates[-1]}")
    got_2023 = any(d.startswith("2023") for d in dates)
    print(f"  contains 2023 dates? {'YES - historical works' if got_2023 else 'NO - only recent (historical may be gated)'}")
    sample = feed[0]
    print(f"  sample overall sentiment: {sample.get('overall_sentiment_score')} "
          f"({sample.get('overall_sentiment_label')})")


def test_stocktwits() -> None:
    print("\n=== StockTwits stream (1 request) ===")
    url = "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json"
    try:
        r = requests.get(url, headers={"User-Agent": C.STOCKTWITS_USER_AGENT}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: request error: {exc}")
        return
    if r.status_code in (401, 403):
        print(f"  BLOCKED: HTTP {r.status_code} - public access restricted. "
              "We would fall back to a labelled financial-sentiment corpus for RQ1.")
        return
    if r.status_code == 429:
        print("  RATE-LIMITED: HTTP 429 - try again later.")
        return
    try:
        r.raise_for_status()
        msgs = r.json().get("messages", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")
        return
    labelled = sum(1 for m in msgs
                   if (m.get("entities") or {}).get("sentiment"))
    print(f"  PASS: {len(msgs)} messages returned, {labelled} carry a Bullish/Bearish tag")
    if msgs:
        sent = (msgs[0].get("entities") or {}).get("sentiment")
        print(f"  sample tag: {sent.get('basic') if sent else 'none'}")


if __name__ == "__main__":
    print("Smoke-testing data sources (one request each)...")
    test_alpha_vantage_news()
    test_stocktwits()
    print("\nDone. Interpret above before running full collectors.")
