"""
Collect daily OHLCV price history from Stooq (free, no API key, ~20 years).

Stooq exposes a simple CSV endpoint:
    https://stooq.com/q/d/l/?s=<ticker>.us&i=d&d1=YYYYMMDD&d2=YYYYMMDD

Output: data/raw/prices/<TICKER>.csv  (columns: Date,Open,High,Low,Close,Volume)

Run:
    python src/collect_prices.py
"""
import sys
import time
import io
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

STOOQ_URL = "https://stooq.com/q/d/l/"


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker's daily OHLCV from Stooq as a DataFrame."""
    params = {
        "s": f"{ticker.lower()}.us",
        "i": "d",
        "d1": start.replace("-", ""),
        "d2": end.replace("-", ""),
    }
    resp = requests.get(STOOQ_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    # Stooq returns the literal string "No data" when a symbol/window is empty.
    if not text or text.lower().startswith("no data") or "Date" not in text:
        raise ValueError(f"Stooq returned no usable data for {ticker}")
    df = pd.read_csv(io.StringIO(text))
    df["Symbol"] = ticker
    return df


def main() -> None:
    ok, failed = [], []
    for ticker in C.TICKERS:
        out = C.RAW_PX / f"{ticker}.csv"
        try:
            df = fetch_one(ticker, C.START_DATE, C.END_DATE)
            df.to_csv(out, index=False)
            print(f"[prices] {ticker:6s} rows={len(df):5d} -> {out.name}")
            ok.append(ticker)
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] {ticker:6s} FAILED: {exc}")
            failed.append(ticker)
        time.sleep(1.0)
    print(f"\nDone. success={len(ok)} failed={len(failed)} {failed if failed else ''}")


if __name__ == "__main__":
    main()
