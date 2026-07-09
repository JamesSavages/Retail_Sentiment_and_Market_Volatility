"""
Collect daily OHLCV price history from Tiingo (free tier: 30+ years of daily
history, ~50 symbols/hour). Split/dividend-ADJUSTED OHLC is used so corporate
actions (e.g., NVDA's 10:1 split in June 2024) do not create fake volatility.

WHY TIINGO: Stooq blocks many IPs/regions, and Alpha Vantage gates full daily
history (outputsize=full) behind a premium plan. Tiingo's free tier serves the
full window cleanly.

Set your free token in .env at the project root:
    TIINGO_API_KEY=your_tiingo_token        # sign up at tiingo.com (free)

Endpoint (per ticker):
    https://api.tiingo.com/tiingo/daily/<ticker>/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&format=csv&token=<TOKEN>

Output: data/raw/prices/<TICKER>.csv  (Date,Open,High,Low,Close,Volume,Symbol)
Resumable: tickers already saved are skipped.

Run:
    uv run src/collect_prices.py
"""
import io
import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)   # read .env, overriding any stale shell variable

BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DELAY_SEC = 1.2              # free tier ~50 symbols/hour; be polite


def fetch_one(ticker: str, token: str) -> pd.DataFrame:
    """Download one ticker's adjusted daily OHLCV over the study window."""
    params = {
        "startDate": C.START_DATE,
        "endDate": C.END_DATE,
        "format": "csv",
        "token": token,
    }
    resp = requests.get(BASE.format(ticker=ticker.lower()), params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text.lower().startswith("date"):   # errors come back as JSON, not CSV
        raise RuntimeError(text[:200])

    df = pd.read_csv(io.StringIO(text))
    # Use split/dividend-adjusted OHLC so corporate actions don't distort volatility.
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]).dt.date,
        "Open": df["adjOpen"],
        "High": df["adjHigh"],
        "Low": df["adjLow"],
        "Close": df["adjClose"],
        "Volume": df["adjVolume"],
        "Symbol": ticker,
    }).sort_values("Date")
    return out


def main() -> None:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        print("ERROR: set TIINGO_API_KEY in your .env (free token at tiingo.com).")
        return

    ok, failed = [], []
    for ticker in C.TICKERS:
        out = C.RAW_PX / f"{ticker}.csv"
        if out.exists():
            print(f"[prices] {ticker:6s} cached, skipping")
            ok.append(ticker)
            continue
        try:
            df = fetch_one(ticker, token)
            if df.empty:
                raise RuntimeError("no rows inside the study window")
            df.to_csv(out, index=False)
            print(f"[prices] {ticker:6s} rows={len(df):5d} -> {out.name}")
            ok.append(ticker)
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] {ticker:6s} FAILED: {exc}")
            failed.append(ticker)
        time.sleep(DELAY_SEC)

    print(f"\nDone. success={len(ok)} failed={len(failed)} {failed if failed else ''}")


if __name__ == "__main__":
    main()