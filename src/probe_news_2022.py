"""
Probe: does Alpha Vantage news sentiment reach back to 2022, and is it dense
enough to be usable?

WHY THIS EXISTS
    Extending the news panel to 2022 costs 96 requests, roughly four days of the
    free tier's 25 per day. This script spends exactly ONE request to find out
    whether that is worth doing, and benchmarks the result against a month
    already collected for 2023.

WHAT IT DOES NOT TOUCH
    The response is written to data/scratch/news_probe/, never to
    data/raw/news/, so build_features.py cannot pick it up.

Run:
    uv run src/probe_news_2022.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)

SCRATCH = C.DATA / "scratch" / "news_probe"
SCRATCH.mkdir(parents=True, exist_ok=True)

AV_URL = "https://www.alphavantage.co/query"
PROBE_SYMBOL = "AAPL"          # widest coverage, so the fairest test of reach
PROBE_MONTH = "202201"
BENCHMARK_MONTH = "202301"     # already collected, for a like for like count


def main() -> None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or key.lower() == "your_key_here":
        print("ERROR: set a real ALPHAVANTAGE_API_KEY in .env")
        return

    out = SCRATCH / f"{PROBE_SYMBOL}_{PROBE_MONTH}.csv"
    if out.exists():
        df = pd.read_csv(out)
        print(f"[probe] cached result reused, no request spent")
    else:
        params = {
            "function": "NEWS_SENTIMENT", "tickers": PROBE_SYMBOL,
            "time_from": f"{PROBE_MONTH}01T0000", "time_to": f"{PROBE_MONTH}31T2359",
            "limit": 1000, "sort": "EARLIEST", "apikey": key,
        }
        print(f"[probe] one request: {PROBE_SYMBOL} {PROBE_MONTH}")
        payload = requests.get(AV_URL, params=params, timeout=60).json()

        if "Note" in payload or "Information" in payload:
            print("  quota or rate limit hit; try again later")
            print("  ", str(payload)[:200])
            return
        if "Error Message" in payload:
            print("  API error:", payload["Error Message"])
            return

        feed = payload.get("feed", [])
        rows = [{"time_published": i.get("time_published"),
                 "title": (i.get("title") or "").replace("\n", " ")[:90]}
                for i in feed]
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)

    n = len(df)
    print(f"\n  articles returned for {PROBE_SYMBOL} {PROBE_MONTH}: {n}")
    if n:
        ts = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
        print(f"  date range   : {ts.min()}  to  {ts.max()}")
        print(f"  distinct days: {ts.dt.date.nunique()}")
        print("\n  earliest three headlines:")
        for t in df["title"].head(3):
            print("   -", t)

    # ---- benchmark against a month already collected --------------------
    bench = C.RAW_NEWS / f"{PROBE_SYMBOL}_{BENCHMARK_MONTH}.csv"
    if bench.exists():
        b = pd.read_csv(bench)
        print(f"\n  benchmark {PROBE_SYMBOL} {BENCHMARK_MONTH}: {len(b)} articles")
        print("\n" + "=" * 60)
        if n == 0:
            print("  VERDICT: no 2022 coverage. Do not spend the quota;")
            print("  record the history limit as a data availability constraint.")
        elif n >= 0.5 * len(b):
            print("  VERDICT: 2022 coverage is comparable to 2023.")
            print("  Extending the window is worth the four days of quota.")
        else:
            print(f"  VERDICT: 2022 coverage is roughly {n / max(len(b), 1):.0%} of 2023.")
            print("  Usable but thinner. Expect more zero-article days, which the")
            print("  news_missing flag already handles, and say so in Limitations.")
        print("=" * 60)
    print(f"\nprobe output written to {out} (safe to delete)")


if __name__ == "__main__":
    main()
