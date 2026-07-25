"""
Assemble the modelling panel (one row per ticker-day) from the raw sources.

For each ticker-day it computes:
  * the PRIMARY TARGET  -> next-day realized volatility  rv_next  (t+1)
  * price-based controls -> log_return, realized volatility (t), log volume
  * HAR components -> rv (daily), rv_w (weekly, 5d), rv_m (monthly, 22d)
  * StockTwits sentiment features (recent sample) -> counts, bullish ratio,
    Antweiler-Frank bullishness and agreement indices, sentiment dispersion
  * Alpha Vantage news sentiment features (historical) -> mean relevance-weighted
    ticker sentiment, article count, zero-attention flag

Volatility is estimated from daily OHLC (no intraday needed) using a range-based
estimator selected in config (Parkinson by default; Garman-Klass or rolling std
also available).

TIMING / LEAKAGE CONVENTION
    rv_t is computed from day t's high and low, so it is observable at the close
    of day t. The target is rv_{t+1}. Every predictor in this panel is therefore
    restricted to information dated <= t, and the HAR components deliberately
    INCLUDE day t (rv_w = mean of rv over t-4..t) rather than being shifted.
    src/validate_panel.py enforces this with a point-in-time reconstruction test.

ZERO-ATTENTION CONVENTION
    A trading day on which Alpha Vantage returned no articles for a ticker is a
    genuine LOW-ATTENTION observation, not a missing one, so it is recorded as
    news_article_count = 0, news_sent_wmean = 0.0 (neutral), news_missing = 1.
    This is applied ONLY to tickers in config.NEWS_TICKERS, i.e. tickers that
    were actually queried. For every other ticker no request was ever made, so
    the news columns stay NaN -- recording a zero there would fabricate a
    negative observation from an absence of collection.

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

# HAR aggregation windows (trading days), inclusive of day t.
HAR_WEEK = 5
HAR_MONTH = 22


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

        # HAR (heterogeneous autoregressive) components. These windows END at t
        # and include it, because rv_t is known at the close of day t while the
        # target is t+1. Corsi's HAR specification is:
        #     rv_{t+1} = b0 + b_d*rv_t + b_w*rv_t^(w) + b_m*rv_t^(m) + e
        df["rv_w"] = df["rv"].rolling(HAR_WEEK, min_periods=HAR_WEEK).mean()
        df["rv_m"] = df["rv"].rolling(HAR_MONTH, min_periods=HAR_MONTH).mean()

        frames.append(df[["symbol", "date", "close", "log_return", "abs_return",
                           "log_volume", "rv", "rv_lag1", "rv_w", "rv_m", "rv_next"]])
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

    # ---- Zero-attention convention (see module docstring) ----------------
    # Only tickers we actually queried can have a *meaningful* zero. For every
    # other ticker the absence of news reflects absence of collection, so the
    # news columns must stay NaN.
    queried = panel["symbol"].isin(C.NEWS_TICKERS)
    no_articles = queried & panel["news_article_count"].isna()

    panel["news_missing"] = np.where(queried, no_articles.astype("int8"), np.nan)
    panel.loc[no_articles, "news_article_count"] = 0
    panel.loc[no_articles, "news_sent_wmean"] = 0.0

    # Attention proxy. log1p is used so that a zero-article day maps to 0 rather
    # than to -inf; this is only well defined now that zero days are retained.
    panel["log_articles"] = np.log1p(panel["news_article_count"])

    n_recovered = int(no_articles.sum())
    print(f"[panel] zero-attention days recorded for {len(C.NEWS_TICKERS)} queried "
          f"tickers: {n_recovered} ticker-days set to article_count=0")

    # Safety net: drop any column that is entirely empty in the current window.
    panel = panel.dropna(axis=1, how="all")

    out = C.PROCESSED / "panel.csv"
    panel.to_csv(out, index=False)
    print(f"[panel] rows={len(panel)} tickers={panel['symbol'].nunique()} -> {out}")
    print(f"[panel] date range: {panel['date'].min()} .. {panel['date'].max()}")
    print(f"[panel] columns: {list(panel.columns)}")

    # ---- Analysis-sample summary (rows usable for RQ2-RQ4) ----
    # A ticker-day is usable when every model input and the target are present.
    # HAR needs rv_m, so the first 21 trading days of each ticker drop out.
    if "news_sent_wmean" not in panel:
        print("\n[analysis-sample] (none yet - run collect_news_sentiment.py)")
        return

    required = ["news_sent_wmean", "log_articles", "rv", "rv_w", "rv_m", "rv_next"]
    usable = panel[panel["symbol"].isin(C.NEWS_TICKERS)].dropna(subset=required)

    # Reported for comparison with the pre-fix figure quoted in earlier drafts.
    strict = usable[usable["news_missing"] == 0]

    print("\n[analysis-sample] RQ2-RQ4 usable ticker-days")
    print(f"  ANALYSIS SAMPLE (incl. zero-attention days) : {len(usable)}")
    print(f"    of which days with >=1 article            : {len(strict)}")
    print(f"    of which zero-attention days              : {len(usable) - len(strict)}")
    print(f"  distinct tickers                            : {usable['symbol'].nunique()}")
    print(f"  distinct trading dates                      : {usable['date'].nunique()}")
    print("\n  per symbol (usable / with-article / zero-attention):")
    for sym in sorted(usable["symbol"].unique()):
        u = usable[usable["symbol"] == sym]
        w = int((u["news_missing"] == 0).sum())
        print(f"    {sym:6s} {len(u):5d} {w:6d} {len(u) - w:6d}")

    print("\n  NOTE: quote the ANALYSIS SAMPLE figure above in the report; the "
          "price panel row count is a separate, larger number.")


if __name__ == "__main__":
    main()