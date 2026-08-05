"""
Probe: is 2022 a genuinely different volatility regime from 2023-2024?

WHY THIS EXISTS
    Extending the study window backwards costs four days of Alpha Vantage news
    quota. That is only worth spending if 2022 actually supplies the second
    volatility regime the current window lacks. This script answers that
    question from prices alone, which are free and unrationed.

WHAT IT DOES NOT TOUCH
    Prices are written to data/scratch/prices_2022/, never to data/raw/prices/.
    build_features.py globs data/raw/prices/, so nothing here can alter the
    modelling panel or the figures behind the submitted report.

Run:
    uv run src/probe_2022_regime.py
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(override=True)

SCRATCH = C.DATA / "scratch" / "prices_2022"
SCRATCH.mkdir(parents=True, exist_ok=True)

BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
PROBE_START, PROBE_END = "2022-01-01", "2022-12-31"
DELAY_SEC = 1.2


def parkinson(h, l):
    """Identical to build_features.parkinson, so the comparison is like for like."""
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(h / l) ** 2))


def fetch(ticker: str, token: str) -> pd.DataFrame:
    params = {"startDate": PROBE_START, "endDate": PROBE_END,
              "format": "csv", "token": token}
    r = requests.get(BASE.format(ticker=ticker.lower()), params=params, timeout=30)
    r.raise_for_status()
    text = r.text.strip()
    if not text.lower().startswith("date"):
        raise RuntimeError(text[:200])
    df = pd.read_csv(io.StringIO(text))
    return pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.tz_localize(None),
        "high": df["adjHigh"], "low": df["adjLow"], "close": df["adjClose"],
        "symbol": ticker,
    }).sort_values("date")


def main() -> None:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        print("ERROR: set TIINGO_API_KEY in .env")
        return

    # ---- collect 2022 prices for the eight sentiment tickers ------------
    frames = []
    for t in C.NEWS_TICKERS:
        out = SCRATCH / f"{t}.csv"
        if out.exists():
            frames.append(pd.read_csv(out, parse_dates=["date"]))
            print(f"[probe] {t:6s} cached")
            continue
        try:
            df = fetch(t, token)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[probe] {t:6s} FAILED: {exc}")
            continue
        df.to_csv(out, index=False)
        frames.append(df)
        print(f"[probe] {t:6s} {len(df):4d} trading days")
        time.sleep(DELAY_SEC)

    if not frames:
        print("no 2022 data retrieved; stopping")
        return

    new = pd.concat(frames, ignore_index=True)
    new["rv"] = parkinson(new["high"], new["low"])

    # ---- existing 2023-2024 panel, restricted to the same eight tickers --
    old = pd.read_csv(C.PROCESSED / "panel.csv", parse_dates=["date"])
    old = old[old["symbol"].isin(C.NEWS_TICKERS)][["symbol", "date", "rv"]].dropna()

    both = pd.concat([new[["symbol", "date", "rv"]], old], ignore_index=True)

    # ---- quarterly cross-sectional mean volatility ----------------------
    q = (both.pivot_table(index="date", columns="symbol", values="rv")
              .mean(axis=1).resample("QE").mean().dropna())

    print("\n" + "=" * 62)
    print("QUARTERLY CROSS-SECTIONAL MEAN REALIZED VOLATILITY")
    print("=" * 62)
    for d, v in q.items():
        bar = "#" * int(round(v * 1200))
        print(f"  {d.date()}  {v:.4f}  {bar}")

    q22 = q[q.index.year == 2022]
    q34 = q[q.index.year.isin((2023, 2024))]
    ratio_all = q.max() / q.min()
    ratio_now = q34.max() / q34.min()

    print("\n" + "-" * 62)
    print(f"  2022 mean quarterly volatility      : {q22.mean():.4f}")
    print(f"  2023-2024 mean quarterly volatility : {q34.mean():.4f}")
    print(f"  2022 is {q22.mean() / q34.mean():.2f}x the 2023-2024 level")
    print(f"\n  highest/lowest quarter, 2023-2024 only : {ratio_now:.2f}x  "
          f"({'single regime' if ratio_now < 2 else 'multiple regimes'})")
    print(f"  highest/lowest quarter, 2022-2024      : {ratio_all:.2f}x  "
          f"({'single regime' if ratio_all < 2 else 'multiple regimes'})")

    print("\n" + "=" * 62)
    if ratio_all >= 2.0:
        print("  VERDICT: adding 2022 breaks the single-regime limitation.")
        print("  Spending the news quota is justified.")
    elif ratio_all >= 1.6:
        print("  VERDICT: 2022 widens the range but does not clear 2x.")
        print("  Defensible, but state the ratio honestly rather than claiming")
        print("  two distinct regimes.")
    else:
        print("  VERDICT: 2022 is not materially more volatile on this measure.")
        print("  Reconsider before spending four days of news quota.")
    print("=" * 62)
    print(f"\nscratch prices written to {SCRATCH} (safe to delete)")


if __name__ == "__main__":
    main()
