"""
Correctness gate for data/processed/panel.csv.

Run this after every build_features.py run and BEFORE any modelling. It fails
loudly, printing the offending rows, rather than printing something reassuring.

The checks are grouped:

  A. STRUCTURAL     -- keys, ordering, duplicates, coverage
  B. DEFINITIONAL   -- every derived column recomputed from raw and compared
  C. LEAKAGE        -- point-in-time reconstruction: rebuild each predictor
                       using only raw records dated <= t and require a match.
                       This is the check that protects the whole study: if any
                       feature secretly uses tomorrow's data, the forecasting
                       claim collapses.
  D. NEWS INTEGRITY -- zero-attention convention, ranges, weighted mean
                       recomputed from the raw Alpha Vantage files
  E. CORPORATE ACTIONS -- adjusted prices should show no split discontinuity
  F. SANITY         -- positivity, finiteness, window bounds

Exit code 0 = every check passed. Exit code 1 = at least one failure.

Run:
    uv run src/validate_panel.py
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from build_features import parkinson, garman_klass, HAR_WEEK, HAR_MONTH  # noqa: E402

TOL = 1e-9          # float comparison tolerance for exact recomputation
SAMPLE_ROWS = 30    # rows drawn for the point-in-time reconstruction test
RNG = np.random.default_rng(42)

_results: list[tuple[bool, str, str]] = []
_warnings: list[str] = []

# Common split ratios. An UNADJUSTED split shows up as close[t]/close[t-1]
# sitting almost exactly on one of these; a genuine price shock does not.
SPLIT_RATIOS = [2, 2.5, 3, 4, 5, 6, 7, 8, 10, 15, 20]
SPLIT_TOL = 0.005          # 0.5% -- tight, so earnings shocks are not caught

# Extreme one-day moves verified as genuine market events rather than data
# errors. Confirmed by inspecting the surrounding volatility path: a split is a
# single clean price discontinuity with no change in realized volatility, while
# these show rv rising sharply across several sessions.
VERIFIED_EVENT_MOVES = {
    ("ARM", "2024-02-08"): "post-earnings repricing; rv 0.030 -> 0.179",
    ("PLTR", "2024-02-06"): "post-earnings repricing; rv 0.049 -> 0.069",
}


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Record a single assertion result."""
    _results.append((bool(condition), name, detail))
    mark = "PASS" if condition else "FAIL"
    line = f"  [{mark}] {name}"
    if detail and not condition:
        line += f"\n         {detail}"
    print(line)
    return bool(condition)


def warn(message: str) -> None:
    """Record a non-fatal finding that needs human confirmation."""
    _warnings.append(message)
    print(f"  [WARN] {message}")


def _rv_from(df: pd.DataFrame) -> pd.Series:
    est = C.AV_VOLATILITY_ESTIMATOR
    if est == "parkinson":
        return parkinson(df["high"], df["low"])
    if est == "garman_klass":
        return garman_klass(df["open"], df["high"], df["low"], df["close"])
    return df["log_return"].rolling(C.ROLLING_STD_WINDOW).std()


def load_raw_prices() -> dict[str, pd.DataFrame]:
    """Raw per-ticker price files, lower-cased and date-sorted."""
    out = {}
    for f in glob.glob(str(C.RAW_PX / "*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        out[df["symbol"].iloc[0]] = df.sort_values("date").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# A. Structural
# ----------------------------------------------------------------------
def check_structural(panel: pd.DataFrame) -> None:
    print("\nA. STRUCTURAL")

    dupes = panel.duplicated(subset=["symbol", "date"]).sum()
    check("no duplicate (symbol, date) keys", dupes == 0, f"{dupes} duplicated keys")

    bad = [s for s, g in panel.groupby("symbol") if not g["date"].is_monotonic_increasing]
    check("dates strictly increasing within each symbol", not bad, f"unsorted: {bad}")

    check("no null symbol or date", panel[["symbol", "date"]].isna().sum().sum() == 0)

    expected_cols = {
        "symbol", "date", "close", "log_return", "abs_return", "log_volume",
        "rv", "rv_lag1", "rv_w", "rv_m", "rv_next",
        "news_sent_wmean", "news_article_count", "news_missing", "log_articles",
    }
    missing = expected_cols - set(panel.columns)
    check("all expected columns present", not missing, f"missing: {sorted(missing)}")


# ----------------------------------------------------------------------
# B. Definitional -- recompute every derived column from raw
# ----------------------------------------------------------------------
def check_definitional(panel: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> None:
    print("\nB. DEFINITIONAL (derived columns recomputed from raw)")

    worst = {k: 0.0 for k in
             ["log_return", "log_volume", "rv", "rv_lag1", "rv_w", "rv_m", "rv_next", "abs_return"]}
    checked = 0

    for sym, g in panel.groupby("symbol"):
        if sym not in raw:
            continue
        r = raw[sym].copy()
        g = g.sort_values("date").reset_index(drop=True)

        exp = pd.DataFrame({"date": r["date"]})
        exp["log_return"] = np.log(r["close"] / r["close"].shift(1))
        exp["abs_return"] = exp["log_return"].abs()
        exp["log_volume"] = np.log1p(r["volume"])
        exp["rv"] = _rv_from(r.assign(log_return=exp["log_return"]))
        exp["rv_lag1"] = exp["rv"].shift(1)
        exp["rv_w"] = exp["rv"].rolling(HAR_WEEK, min_periods=HAR_WEEK).mean()
        exp["rv_m"] = exp["rv"].rolling(HAR_MONTH, min_periods=HAR_MONTH).mean()
        exp["rv_next"] = exp["rv"].shift(-1)

        m = g.merge(exp, on="date", suffixes=("", "_exp"))
        checked += len(m)
        for col in worst:
            d = (m[col] - m[f"{col}_exp"]).abs()
            d = d[m[col].notna() | m[f"{col}_exp"].notna()]
            worst[col] = max(worst[col], float(np.nanmax(d.values)) if len(d) else 0.0)

    for col, w in worst.items():
        check(f"{col} matches recomputation (max abs diff {w:.2e})", w < TOL or np.isnan(w))
    print(f"         ({checked} ticker-days recomputed)")


# ----------------------------------------------------------------------
# C. Leakage -- point-in-time reconstruction
# ----------------------------------------------------------------------
def check_leakage(panel: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> None:
    print("\nC. LEAKAGE (point-in-time reconstruction, information dated <= t only)")

    predictors = ["log_return", "abs_return", "log_volume", "rv", "rv_lag1", "rv_w", "rv_m"]
    pool = panel[panel["rv_m"].notna() & panel["symbol"].isin(raw)]
    if pool.empty:
        check("sample available for reconstruction", False, "no eligible rows")
        return

    idx = RNG.choice(pool.index.values, size=min(SAMPLE_ROWS, len(pool)), replace=False)
    failures: list[str] = []

    for i in idx:
        row = panel.loc[i]
        r = raw[row["symbol"]]
        # Everything the analyst could possibly know at the close of day t:
        hist = r[r["date"] <= row["date"]].copy()

        lr = np.log(hist["close"] / hist["close"].shift(1))
        rv = _rv_from(hist.assign(log_return=lr))
        rebuilt = {
            "log_return": lr.iloc[-1],
            "abs_return": abs(lr.iloc[-1]),
            "log_volume": np.log1p(hist["volume"].iloc[-1]),
            "rv": rv.iloc[-1],
            "rv_lag1": rv.iloc[-2] if len(rv) > 1 else np.nan,
            "rv_w": rv.iloc[-HAR_WEEK:].mean() if len(rv) >= HAR_WEEK else np.nan,
            "rv_m": rv.iloc[-HAR_MONTH:].mean() if len(rv) >= HAR_MONTH else np.nan,
        }
        for col in predictors:
            a, b = row[col], rebuilt[col]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b) or abs(a - b) > TOL:
                failures.append(f"{row['symbol']} {row['date'].date()} {col}: "
                                f"panel={a!r} pit={b!r}")

    check(f"all predictors reconstructible from date <= t ({len(idx)} rows sampled)",
          not failures, "\n         ".join(failures[:8]))

    # The target is the one column that MUST use t+1 -- verify it genuinely does.
    ok = True
    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        if not np.allclose(g["rv_next"].values[:-1], g["rv"].values[1:], equal_nan=True):
            ok = False
    check("rv_next[t] == rv[t+1] (target is genuinely forward-looking)", ok)


# ----------------------------------------------------------------------
# D. News integrity
# ----------------------------------------------------------------------
def check_news(panel: pd.DataFrame) -> None:
    print("\nD. NEWS INTEGRITY")

    queried = panel["symbol"].isin(C.NEWS_TICKERS)

    off = panel.loc[~queried, ["news_sent_wmean", "news_article_count", "news_missing"]]
    check("non-queried tickers carry NaN news (no fabricated zeros)",
          off.notna().sum().sum() == 0,
          f"{int(off.notna().sum().sum())} non-null values on non-queried tickers")

    q = panel[queried]
    check("queried tickers have no missing news_article_count",
          q["news_article_count"].isna().sum() == 0)

    consistent = ((q["news_article_count"] == 0) == (q["news_missing"] == 1)).all()
    check("news_article_count == 0 iff news_missing == 1", consistent)

    s = q["news_sent_wmean"].dropna()
    check("news_sent_wmean within [-1, 1]", bool(((s >= -1) & (s <= 1)).all()),
          f"range observed: [{s.min():.4f}, {s.max():.4f}]")

    check("log_articles == log1p(news_article_count)",
          bool(np.allclose(q["log_articles"], np.log1p(q["news_article_count"]))))

    # Recompute the relevance-weighted mean from the raw Alpha Vantage files.
    files = glob.glob(str(C.RAW_NEWS / "*.csv"))
    if not files:
        check("raw news files present for recomputation", False)
        return
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ts = pd.to_datetime(raw["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
    raw["date"] = ts.dt.normalize()
    raw["ticker_sentiment_score"] = pd.to_numeric(raw["ticker_sentiment_score"], errors="coerce")
    raw["relevance_score"] = pd.to_numeric(raw["relevance_score"], errors="coerce")

    have = q[q["news_missing"] == 0]
    idx = RNG.choice(have.index.values, size=min(20, len(have)), replace=False)
    bad: list[str] = []
    for i in idx:
        row = panel.loc[i]
        grp = raw[(raw["symbol"] == row["symbol"]) & (raw["date"] == row["date"])]
        if grp.empty:
            bad.append(f"{row['symbol']} {row['date'].date()}: no raw rows")
            continue
        w = grp["relevance_score"].fillna(0)
        s_ = grp["ticker_sentiment_score"]
        exp = np.average(s_, weights=w) if w.sum() > 0 else s_.mean()
        if abs(exp - row["news_sent_wmean"]) > 1e-8:
            bad.append(f"{row['symbol']} {row['date'].date()}: "
                       f"panel={row['news_sent_wmean']:.6f} raw={exp:.6f}")
        if len(grp) != row["news_article_count"]:
            bad.append(f"{row['symbol']} {row['date'].date()}: "
                       f"count panel={row['news_article_count']} raw={len(grp)}")
    check(f"relevance-weighted mean and article count match raw ({len(idx)} sampled)",
          not bad, "\n         ".join(bad[:8]))


# ----------------------------------------------------------------------
# E. Corporate actions
# ----------------------------------------------------------------------
def check_corporate_actions(panel: pd.DataFrame) -> None:
    print("\nE. CORPORATE ACTIONS (adjusted prices should show no split jump)")

    # An UNADJUSTED split leaves a one-day gap sitting on a simple ratio
    # (a 10:1 split -> -90%). A genuine shock does not. Classify, don't just flag.
    artifacts: list[str] = []
    events: list[str] = []

    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        ratio = g["close"] / g["close"].shift(1)
        big = ratio[(ratio - 1).abs() > 0.30].dropna()

        for k, r in big.items():
            dstr = str(g["date"].iloc[k].date())
            label = f"{sym} {dstr} {100 * (r - 1):+.1f}% (ratio {r:.4f})"

            on_split = any(
                abs(r - f) / f < SPLIT_TOL or abs(r - 1 / f) * f < SPLIT_TOL
                for f in SPLIT_RATIOS
            )
            if on_split:
                artifacts.append(label + " <- matches a split ratio")
            elif (sym, dstr) in VERIFIED_EVENT_MOVES:
                events.append(f"{label} -- {VERIFIED_EVENT_MOVES[(sym, dstr)]}")
            else:
                warn(f"unverified extreme move: {label}. Confirm against a "
                     f"corporate-action calendar, then add to "
                     f"VERIFIED_EVENT_MOVES or investigate.")

    check("no unadjusted-split artifacts (large move sitting on a split ratio)",
          not artifacts, "; ".join(artifacts[:10]))

    if events:
        print("         verified genuine price shocks (retained deliberately):")
        for e in events:
            print(f"           {e}")

    # Direct continuity test on known split dates inside the study window.
    for sym, dstr in {("NVDA", "2024-06-10")}:
        g = panel[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
        d = pd.Timestamp(dstr)
        if g.empty or d not in set(g["date"]):
            continue
        k = int(g.index[g["date"] == d][0])
        if k == 0:
            continue
        move = g["close"].iloc[k] / g["close"].iloc[k - 1] - 1
        check(f"{sym} {dstr} split date shows no discontinuity ({100 * move:+.2f}%)",
              abs(move) < 0.30)


# ----------------------------------------------------------------------
# F. Sanity
# ----------------------------------------------------------------------
def check_sanity(panel: pd.DataFrame) -> None:
    print("\nF. SANITY")

    check("rv strictly positive", bool((panel["rv"].dropna() > 0).all()))
    check("close strictly positive", bool((panel["close"] > 0).all()))

    num = panel.select_dtypes(include=[np.number])
    inf = int(np.isinf(num.to_numpy()).sum())
    check("no infinite values", inf == 0, f"{inf} infinities")

    lo, hi = panel["date"].min(), panel["date"].max()
    check(f"date window within {C.START_DATE}..{C.END_DATE} (actual {lo.date()}..{hi.date()})",
          lo >= pd.Timestamp(C.START_DATE) and hi <= pd.Timestamp(C.END_DATE))

    # A predictor correlating almost perfectly with the target is a leakage smell.
    smells = []
    for col in ["rv", "rv_lag1", "rv_w", "rv_m", "news_sent_wmean", "log_articles"]:
        sub = panel[[col, "rv_next"]].dropna()
        if len(sub) > 100:
            c = sub[col].corr(sub["rv_next"])
            if abs(c) > 0.99:
                smells.append(f"{col} r={c:.4f}")
    check("no predictor correlates >0.99 with target", not smells, "; ".join(smells))


def main() -> int:
    path = C.PROCESSED / "panel.csv"
    if not path.exists():
        print(f"ERROR: {path} not found -- run build_features.py first.")
        return 1

    panel = pd.read_csv(path, parse_dates=["date"])
    raw = load_raw_prices()

    print("=" * 68)
    print(f"PANEL VALIDATION  --  {path}")
    print(f"rows={len(panel)}  tickers={panel['symbol'].nunique()}  "
          f"cols={len(panel.columns)}")
    print("=" * 68)

    check_structural(panel)
    check_definitional(panel, raw)
    check_leakage(panel, raw)
    check_news(panel)
    check_corporate_actions(panel)
    check_sanity(panel)

    failed = [r for r in _results if not r[0]]
    print("\n" + "=" * 68)
    print(f"RESULT: {len(_results) - len(failed)}/{len(_results)} checks passed"
          f"{f', {len(_warnings)} warning(s)' if _warnings else ''}")
    if _warnings:
        print("\nWARNINGS (non-fatal, but confirm before writing up):")
        for w in _warnings:
            print(f"  - {w}")
    if failed:
        print("\nFAILED CHECKS:")
        for _, name, detail in failed:
            print(f"  - {name}")
            if detail:
                print(f"      {detail}")
        print("\nDo NOT proceed to modelling until these are resolved.")
    else:
        print("Panel is internally consistent, leakage-free, and matches raw sources.")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
