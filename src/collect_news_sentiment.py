"""
Collect historical per-ticker news sentiment from Alpha Vantage (NEWS_SENTIMENT).
Sentiment FEATURE source for the 2023-2024 panel (RQ2-RQ4).

Free tier = 25 requests/day, so this paginates by month and CACHES each
(ticker, month). Re-running skips months already downloaded, letting you
accumulate the panel across several days without wasting quota.

Robust to transient network timeouts: each request is retried with backoff, and
a month that still fails is skipped (you recover it on the next resume run)
rather than crashing the whole job.

Set your free key in .env at the project root:
    ALPHAVANTAGE_API_KEY=your_key

Output: data/raw/news/<TICKER>_<YYYYMM>.csv
Run:
    uv run src/collect_news_sentiment.py
"""
import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)  # read .env, overriding any stale shell variable

AV_URL = "https://www.alphavantage.co/query"
RETRY_BACKOFF = [5, 15, 30, 45]   # seconds; len == max retry attempts


def get_with_retries(params: dict) -> requests.Response:
    """GET with retry/backoff on transient network errors (timeouts, resets)."""
    last_exc = None
    for attempt, wait in enumerate(RETRY_BACKOFF):
        try:
            return requests.get(AV_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(f"    network issue ({type(exc).__name__}); "
                  f"retry {attempt + 1}/{len(RETRY_BACKOFF)} in {wait}s")
            time.sleep(wait)
    raise last_exc


def month_windows(start: str, end: str):
    rng = pd.date_range(start=start, end=end, freq="MS")
    for m in rng:
        m_end = m + pd.offsets.MonthEnd(1)
        yield m.strftime("%Y%m"), m.strftime("%Y%m%dT0000"), m_end.strftime("%Y%m%dT2359")


def parse_feed(payload: dict, symbol: str) -> pd.DataFrame:
    rows = []
    for item in payload.get("feed", []):
        t_score, t_rel = None, None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == symbol:
                t_score = ts.get("ticker_sentiment_score")
                t_rel = ts.get("relevance_score")
                break
        rows.append({
            "time_published": item.get("time_published"),
            "symbol": symbol,
            "title": (item.get("title") or "").replace("\n", " ").strip(),
            "source": item.get("source"),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
            "ticker_sentiment_score": t_score,
            "relevance_score": t_rel,
        })
    return pd.DataFrame(rows)


def main() -> None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or key.lower() == "your_key_here":
        print("ERROR: set a real ALPHAVANTAGE_API_KEY in .env "
              "(free key at https://www.alphavantage.co/support/#api-key).")
        return

    requests_made = 0
    for symbol in C.NEWS_TICKERS:
        for ym, t_from, t_to in month_windows(C.START_DATE, C.END_DATE):
            out = C.RAW_NEWS / f"{symbol}_{ym}.csv"
            if out.exists():
                continue  # cached; protect the daily quota
            params = {
                "function": "NEWS_SENTIMENT", "tickers": symbol,
                "time_from": t_from, "time_to": t_to,
                "limit": 1000, "sort": "EARLIEST", "apikey": key,
            }
            try:
                resp = get_with_retries(params)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                print(f"[news] {symbol:6s} {ym} FAILED after retries: {exc} "
                      "-- skipping; re-run later to resume this month.")
                continue

            requests_made += 1
            if "Note" in payload or "Information" in payload:
                print(f"[news] quota/limit reached after {requests_made} requests: "
                      f"{payload.get('Note') or payload.get('Information')}")
                print("Re-run tomorrow; cached months are skipped automatically.")
                return

            df = parse_feed(payload, symbol)
            df.to_csv(out, index=False)
            print(f"[news] {symbol:6s} {ym} articles={len(df):4d} -> {out.name}")
            time.sleep(C.AV_DELAY_SEC)

    print(f"\nDone. requests_made={requests_made}")


if __name__ == "__main__":
    main()