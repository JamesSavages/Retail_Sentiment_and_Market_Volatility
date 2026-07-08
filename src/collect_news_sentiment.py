"""
Collect historical per-ticker news sentiment from Alpha Vantage (NEWS_SENTIMENT).
This is the sentiment FEATURE source for the 2023-2024 panel (RQ2-RQ4), because
the free StockTwits stream cannot reach that far back.

Free tier = 25 requests/day, so this script paginates by month and caches each
(ticker, month) response. Re-running it skips months already downloaded, letting
you accumulate the panel across several days without wasting quota.

Set your free key first:
    export ALPHAVANTAGE_API_KEY=xxxxxxxx      # get one at alphavantage.co (free)

Endpoint:
    https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=<T>
        &time_from=YYYYMMDDT0000&time_to=YYYYMMDDT2359&limit=1000&apikey=<KEY>

Output: data/raw/news/<TICKER>_<YYYYMM>.csv  (one row per article)
    columns: time_published, symbol, title, source, overall_sentiment_score,
             overall_sentiment_label, ticker_sentiment_score, relevance_score

Run:
    python src/collect_news_sentiment.py
"""
import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

AV_URL = "https://www.alphavantage.co/query"


def month_windows(start: str, end: str):
    """Yield (YYYYMM, time_from, time_to) monthly windows across [start, end]."""
    rng = pd.date_range(start=start, end=end, freq="MS")
    for m in rng:
        m_end = (m + pd.offsets.MonthEnd(1))
        yield m.strftime("%Y%m"), m.strftime("%Y%m%dT0000"), m_end.strftime("%Y%m%dT2359")


def parse_feed(payload: dict, symbol: str) -> pd.DataFrame:
    rows = []
    for item in payload.get("feed", []):
        # find this ticker's sentiment inside the article's ticker_sentiment list
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
    if not key:
        print("ERROR: set ALPHAVANTAGE_API_KEY in your environment (free key at "
              "https://www.alphavantage.co/support/#api-key).")
        return

    requests_made = 0
    for symbol in C.NEWS_TICKERS:
        for ym, t_from, t_to in month_windows(C.START_DATE, C.END_DATE):
            out = C.RAW_NEWS / f"{symbol}_{ym}.csv"
            if out.exists():
                continue  # cached; skip to protect the daily quota
            params = {
                "function": "NEWS_SENTIMENT", "tickers": symbol,
                "time_from": t_from, "time_to": t_to,
                "limit": 1000, "sort": "EARLIEST", "apikey": key,
            }
            resp = requests.get(AV_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            requests_made += 1

            # Alpha Vantage signals quota exhaustion via a "Note"/"Information" field.
            if "Note" in payload or "Information" in payload:
                print(f"[news] quota/limit hit after {requests_made} requests: "
                      f"{payload.get('Note') or payload.get('Information')}")
                print("Re-run tomorrow; cached months will be skipped.")
                return

            df = parse_feed(payload, symbol)
            df.to_csv(out, index=False)
            print(f"[news] {symbol:6s} {ym} articles={len(df):4d} -> {out.name}")
            time.sleep(C.AV_DELAY_SEC)

    print(f"\nDone. requests_made={requests_made}")


if __name__ == "__main__":
    main()
