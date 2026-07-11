"""
Assemble the modelling panel (one row per ticker-day) from the raw sources.

For each ticker-day it computes:
  * the PRIMARY TARGET  -> next-day realized volatility  rv_next  (t+1)
  * price-based controls -> log_return, realized volatility (t), log volume
  * StockTwits sentiment features (recent sample) -> counts, bullish ratio,
    Antweiler-Frank bullishness and agreement indices, sentiment dispersion
  * Alpha Vantage news sentiment features (historical) -> mean relevance-weighted
    ticker sentiment, article count

Volatility is estimated from daily OHLC (no intraday needed) using a range-based
estimator selected in config (Parkinson by default; Garman-Klass or rolling std
also available).

Output: data/processed/panel.csv

Run (after the collectors):
    python src/build_features.py
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


# ----------------------------------------------------------------------
# Volatility estimators (daily, from OHLC)
# ----------------------------------------------------------------------
def parkinson(h, l):
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(h / l) ** 2))


def garman_klass(o, h, l, c):
    return np.sqrt(0.5 * (np.log(h / l) ** 2) - (2 * np.log(2) - 1) * (np.log(c / o) ** 2))


# ----------------------------------------------------------------------
# Price panel: target + controls
# ----------------------------------------------------------------------
def build_price_panel() -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(C.RAW_PX / "*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date")
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["log_volume"] = np.log1p(df["volume"])

        est = C.AV_VOLATILITY_ESTIMATOR
        if est == "parkinson":
            df["rv"] = parkinson(df["high"], df["low"])
        elif est == "garman_klass":
            df["rv"] = garman_klass(df["open"], df["high"], df["low"], df["close"])
        else:  # rolling_std of log returns
            df["rv"] = df["log_return"].rolling(C.ROLLING_STD_WINDOW).std()

        # predictive target: next trading day's realized volatility (t+1)
        df["rv_next"] = df["rv"].shift(-1)
        # lagged controls (what the market already knew at t)
        df["rv_lag1"] = df["rv"].shift(1)
        df["abs_return"] = df["log_return"].abs()
        frames.append(df[["symbol", "date", "close", "log_return", "abs_return",
                           "log_volume", "rv", "rv_lag1", "rv_next"]])
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# StockTwits sentiment aggregation (recent sample -> RQ1 features)
# ----------------------------------------------------------------------
def _af_bullishness(n_bull, n_bear):
    return np.log((1.0 + n_bull) / (1.0 + n_bear))


def _af_agreement(n_bull, n_bear):
    n = n_bull + n_bear
    if n == 0:
        return 0.0
    return 1.0 - np.sqrt(1.0 - ((n_bull - n_bear) / n) ** 2)


def build_stocktwits_panel() -> pd.DataFrame:
    files = glob.glob(str(C.RAW_ST / "*.csv"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date"])
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        ts = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        et = ts.dt.tz_convert("US/Eastern")
        trade_date = et.dt.normalize()
        if C.ROLL_POST_CLOSE_TO_NEXT_DAY:
            after_close = et.dt.hour >= 16
            trade_date = trade_date + pd.to_timedelta(after_close.astype(int), unit="D")
        df["date"] = trade_date.dt.date
        df["is_bull"] = df["sentiment_basic"].eq("Bullish")
        df["is_bear"] = df["sentiment_basic"].eq("Bearish")
        g = df.groupby(["symbol", "date"]).agg(
            st_n_msgs=("id", "count"),
            st_n_bull=("is_bull", "sum"),
            st_n_bear=("is_bear", "sum"),
        ).reset_index()
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["st_labelled"] = out["st_n_bull"] + out["st_n_bear"]
    out["st_bullish_ratio"] = out["st_n_bull"] / out["st_labelled"].replace(0, np.nan)
    out["st_labelled_share"] = out["st_labelled"] / out["st_n_msgs"]
    out["st_log_msgs"] = np.log1p(out["st_n_msgs"])
    out["st_af_bullishness"] = [_af_bullishness(b, r) for b, r in zip(out["st_n_bull"], out["st_n_bear"])]
    out["st_af_agreement"] = [_af_agreement(b, r) for b, r in zip(out["st_n_bull"], out["st_n_bear"])]
    return out


# ----------------------------------------------------------------------
# Alpha Vantage news sentiment aggregation (historical -> RQ2-4 features)
# ----------------------------------------------------------------------
def build_news_panel() -> pd.DataFrame:
    files = glob.glob(str(C.RAW_NEWS / "*.csv"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    ts = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["date"] = ts.dt.date
    df["ticker_sentiment_score"] = pd.to_numeric(df["ticker_sentiment_score"], errors="coerce")
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")

    def wmean(group):
        w = group["relevance_score"].fillna(0)
        s = group["ticker_sentiment_score"]
        return np.average(s, weights=w) if w.sum() > 0 else s.mean()

    rows = []
    for (sym, d), grp in df.groupby(["symbol", "date"]):
        rows.append({
            "symbol": sym, "date": d,
            "news_sent_wmean": wmean(grp),
            "news_article_count": len(grp),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------
def main() -> None:
    # The historical RQ2-RQ4 panel is built from PRICES + NEWS only.
    # StockTwits serves recent messages only, so it cannot supply features over
    # the 2023-2024 window; it is the RQ1 labelled corpus, analysed at the
    # message level directly from data/raw/stocktwits/ and intentionally NOT
    # merged here. (build_stocktwits_panel remains available for a future
    # recent-window extension.)
    price = build_price_panel()
    news = build_news_panel()

    panel = price.merge(news, on=["symbol", "date"], how="left")
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # Safety net: drop any column that is entirely empty in the current window.
    panel = panel.dropna(axis=1, how="all")

    out = C.PROCESSED / "panel.csv"
    panel.to_csv(out, index=False)
    print(f"[panel] rows={len(panel)} tickers={panel['symbol'].nunique()} -> {out}")
    print(f"[panel] date range: {panel['date'].min()} .. {panel['date'].max()}")
    print(f"[panel] columns: {list(panel.columns)}")


if __name__ == "__main__":
    main()