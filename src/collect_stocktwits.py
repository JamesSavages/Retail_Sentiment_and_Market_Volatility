"""
Collect RECENT StockTwits messages (with users' self-tagged Bullish/Bearish
labels) for each ticker. This labelled corpus is used to train and benchmark
the RQ1 sentiment model (FinBERT vs. lexicon vs. the self-tags).

NOTE ON ACCESS
--------------
The public stream endpoint returns only recent messages and is rate limited;
StockTwits has tightened access over time. This script handles 401/403/429
gracefully. If the public endpoint is blocked for you, either request a
StockTwits API token / partner access, or rely on Alpha Vantage news sentiment
(collect_news_sentiment.py) for the historical feature and use any small
labelled StockTwits sample you can obtain for RQ1 validation.

Endpoint:
    https://api.stocktwits.com/api/2/streams/symbol/<SYMBOL>.json?max=<message_id>

Output: data/raw/stocktwits/<TICKER>.csv
    columns: id, created_at, symbol, body, sentiment_basic, user_id, user_followers

Run:
    python src/collect_stocktwits.py
"""
import sys
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

BASE = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def _parse_messages(payload: dict, symbol: str) -> list[dict]:
    rows = []
    for m in payload.get("messages", []):
        sentiment = None
        entities = m.get("entities") or {}
        sent = entities.get("sentiment")
        if isinstance(sent, dict):
            sentiment = sent.get("basic")  # "Bullish" / "Bearish"
        user = m.get("user") or {}
        rows.append({
            "id": m.get("id"),
            "created_at": m.get("created_at"),
            "symbol": symbol,
            "body": (m.get("body") or "").replace("\n", " ").strip(),
            "sentiment_basic": sentiment,
            "user_id": user.get("id"),
            "user_followers": user.get("followers"),
        })
    return rows


def fetch_ticker(symbol: str, pages: int) -> pd.DataFrame:
    rows, max_id = [], None
    for _ in range(pages):
        params = {"max": max_id} if max_id else {}
        resp = requests.get(
            BASE.format(symbol=symbol),
            params=params,
            headers={"User-Agent": C.STOCKTWITS_USER_AGENT},
            timeout=30,
        )
        if resp.status_code in (401, 403):
            print(f"[stocktwits] {symbol}: access restricted ({resp.status_code}). "
                  "Need a token/partner access - see module docstring.")
            break
        if resp.status_code == 429:
            print(f"[stocktwits] {symbol}: rate limited (429). Backing off 60s.")
            time.sleep(60)
            continue
        resp.raise_for_status()
        payload = resp.json()
        page_rows = _parse_messages(payload, symbol)
        if not page_rows:
            break
        rows.extend(page_rows)
        max_id = min(r["id"] for r in page_rows) - 1
        time.sleep(C.STOCKTWITS_DELAY_SEC)
    return pd.DataFrame(rows)


def main() -> None:
    for symbol in C.TICKERS:
        out = C.RAW_ST / f"{symbol}.csv"
        try:
            df = fetch_ticker(symbol, C.STOCKTWITS_PAGES_PER_TICKER)
            if len(df):
                df.to_csv(out, index=False)
                labelled = df["sentiment_basic"].notna().sum()
                print(f"[stocktwits] {symbol:6s} msgs={len(df):4d} labelled={labelled:4d} -> {out.name}")
            else:
                print(f"[stocktwits] {symbol:6s} no messages returned")
        except Exception as exc:  # noqa: BLE001
            print(f"[stocktwits] {symbol:6s} FAILED: {exc}")


if __name__ == "__main__":
    main()
