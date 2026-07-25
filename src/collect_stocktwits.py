"""
Collect RECENT StockTwits messages (with users' self-tagged Bullish/Bearish
labels) for each ticker. This labelled corpus trains and benchmarks the RQ1
sentiment model (FinBERT vs. lexicon vs. the self-tags).

ACCUMULATES ACROSS RUNS: StockTwits only serves recent messages, so each run is
merged into the existing per-ticker CSV and de-duplicated by message id. Running
it repeatedly (daily, or whenever the connection is good) steadily grows the
labelled set instead of overwriting it.

NOTE ON ACCESS: the public stream is recent-only and rate limited; it may return
401/403 if access is restricted. This script handles that gracefully.

Endpoint:
    https://api.stocktwits.com/api/2/streams/symbol/<SYMBOL>.json?max=<message_id>

PRIVACY: no user-identifying fields are retained. The API returns a user object
(id, username, follower count), but persisting it would allow individual users to
be profiled, which the study's ethics statement rules out. Only the message id,
timestamp, symbol, body text and the self-tagged sentiment label are stored, and
all downstream analysis is either message-level-anonymous or aggregated to the
ticker-day. See also strip_user_fields() for cleaning any legacy files.

Output: data/raw/stocktwits/<TICKER>.csv
    columns: id, created_at, symbol, body, sentiment_basic

Run (safe to re-run; it merges, not overwrites):
    uv run src/collect_stocktwits.py
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
        # The API also returns m["user"] (id, username, followers). It is
        # deliberately NOT captured -- see the PRIVACY note in the module
        # docstring. Do not reinstate these fields.
        rows.append({
            "id": m.get("id"),
            "created_at": m.get("created_at"),
            "symbol": symbol,
            "body": (m.get("body") or "").replace("\n", " ").strip(),
            "sentiment_basic": sentiment,
        })
    return rows


# Fields that must never persist to disk (legacy files may still contain them).
USER_FIELDS = ["user_id", "user_followers", "username", "user_name"]


def strip_user_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any user-identifying columns inherited from earlier collection runs."""
    drop = [c for c in df.columns if c in USER_FIELDS]
    return df.drop(columns=drop) if drop else df


def fetch_ticker(symbol: str, pages: int) -> pd.DataFrame:
    rows, max_id = [], None
    for _ in range(pages):
        params = {"max": max_id} if max_id else {}
        try:
            resp = requests.get(
                BASE.format(symbol=symbol), params=params,
                headers={"User-Agent": C.STOCKTWITS_USER_AGENT}, timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            print(f"    network issue ({type(exc).__name__}); stopping this ticker early")
            break
        if resp.status_code in (401, 403):
            print(f"[stocktwits] {symbol}: access restricted ({resp.status_code}).")
            break
        if resp.status_code == 429:
            print(f"[stocktwits] {symbol}: rate limited (429). Backing off 60s.")
            time.sleep(60)
            continue
        resp.raise_for_status()
        page_rows = _parse_messages(resp.json(), symbol)
        if not page_rows:
            break
        rows.extend(page_rows)
        max_id = min(r["id"] for r in page_rows) - 1
        time.sleep(C.STOCKTWITS_DELAY_SEC)
    return pd.DataFrame(rows)


def merge_and_save(symbol: str, new_df: pd.DataFrame) -> tuple[int, int]:
    """Union new messages with any existing file; de-dup by id. Returns (total, labelled)."""
    out = C.RAW_ST / f"{symbol}.csv"
    if out.exists():
        old = strip_user_fields(pd.read_csv(out))
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = strip_user_fields(combined)
    combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    combined.to_csv(out, index=False)
    labelled = combined["sentiment_basic"].notna().sum()
    return len(combined), int(labelled)


def main() -> None:
    for symbol in C.TICKERS:
        try:
            new_df = fetch_ticker(symbol, C.STOCKTWITS_PAGES_PER_TICKER)
            if new_df.empty and not (C.RAW_ST / f"{symbol}.csv").exists():
                print(f"[stocktwits] {symbol:6s} no messages returned")
                continue
            total, labelled = merge_and_save(symbol, new_df)
            print(f"[stocktwits] {symbol:6s} new={len(new_df):4d}  "
                  f"total_unique={total:5d}  labelled={labelled:5d}")
        except Exception as exc:  # noqa: BLE001
            print(f"[stocktwits] {symbol:6s} FAILED: {exc}")


if __name__ == "__main__":
    main()