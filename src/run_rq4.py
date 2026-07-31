"""
RQ4: Does a sentiment-informed strategy beat buy-and-hold on risk-adjusted
return, net of transaction costs?

    H0: Sharpe(strategy) <= Sharpe(buy-and-hold)
    H1: Sharpe(strategy) >  Sharpe(buy-and-hold)
    Test: Ledoit-Wolf (2008) robust Sharpe comparison, implemented with a
          circular block bootstrap so that volatility clustering and serial
          dependence are preserved in the resampling.

THE RULE IS FIXED IN ADVANCE
    Specifying the exposure rule before seeing any performance figure is what
    makes this a test rather than a search. Trying several rules and reporting
    the best would invalidate the p-value. The primary rule, the robustness
    rule, and both cost levels were committed to before this script was run;
    only turnover, a mechanical property of each rule, was inspected beforehand.

    PRIMARY   (B) Volatility targeting. Each ticker's weight is
                  target / forecast, capped at 1.0, where target is the
                  expanding median of that ticker's own prior forecasts.
    ROBUSTNESS(A) Threshold de-risking. Weight 0.5 when the forecast exceeds
                  the expanding 80th percentile of that ticker's own prior
                  forecasts, else 1.0.

    Both thresholds are expanding-window and per ticker, so no future
    information enters the exposure decision.

BENCHMARK
    Equal-weight buy-and-hold across the same eight tickers over the same
    out-of-sample window, with one initial purchase cost.

COSTS
    Charged on |change in weight| at 5 and 10 basis points. Ten bp is
    deliberately conservative for mega-cap US equities traded at the close,
    where spreads are typically 1-2 bp and retail commission is zero.

RETURNS
    Measured as excess over a constant 5% annual risk-free rate, roughly the
    average US three-month Treasury yield across 2023-2024. Capital not
    allocated to equities earns the risk-free rate, so its excess return is
    zero -- the standard treatment, and it avoids penalising the strategy for
    holding cash during a high-rate period.

Outputs:
    docs/tables/rq4_performance.csv   headline strategy vs benchmark metrics
    docs/tables/rq4_tests.csv         Ledoit-Wolf bootstrap results
    docs/figures/fig_rq4_backtest.png cumulative excess return and exposure

Run:
    uv run src/run_rq4.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import statsmodels.api as sm  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

FIG_DIR = C.ROOT / "docs" / "figures"
TAB_DIR = C.ROOT / "docs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HAR = ["rv", "rv_w", "rv_m", "log_volume"]
SENT = ["news_sent_wmean", "log_articles", "news_missing"]
N_FOLDS, MIN_TRAIN_FRAC = 6, 0.40
RF_ANNUAL = 0.05
MIN_HIST = 40                 # days before an expanding threshold is trusted
COSTS_BP = [5, 10]
BLOCK, N_BOOT, SEED = 10, 5000, 42
BLUE, GREY, RED = "#2c6fbb", "#4d4d4d", "#c0392b"

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ----------------------------------------------------------------------
# Walk-forward forecasts from the RQ2 augmented model
# ----------------------------------------------------------------------
def forecasts() -> pd.DataFrame:
    p = pd.read_csv(C.PROCESSED / "panel.csv", parse_dates=["date"])
    d = p[p["symbol"].isin(C.NEWS_TICKERS)].dropna(
        subset=HAR + SENT + ["rv_next", "log_return"]).copy()
    d["y"] = np.log(d["rv_next"])
    d = pd.concat([d, pd.get_dummies(d["symbol"], prefix="fe",
                                     drop_first=True, dtype=float)], axis=1)
    d = d.sort_values(["date", "symbol"]).reset_index(drop=True)
    cols = HAR + SENT + [c for c in d.columns if c.startswith("fe_")]

    dates = np.sort(d["date"].unique())
    edges = np.linspace(int(len(dates) * MIN_TRAIN_FRAC), len(dates), N_FOLDS + 1).astype(int)
    d["fc"] = np.nan
    for i in range(N_FOLDS):
        tr = d["date"] < dates[edges[i]]
        te = (d["date"] >= dates[edges[i]]) & (
            (d["date"] < dates[edges[i + 1]]) if edges[i + 1] < len(dates) else True)
        X = sm.add_constant(d.loc[tr, cols].astype(float), has_constant="add")
        m = sm.OLS(d.loc[tr, "y"], X).fit()
        Xte = sm.add_constant(d.loc[te, cols].astype(float), has_constant="add")[X.columns]
        d.loc[te, "fc"] = np.exp(m.predict(Xte))

    d = d.dropna(subset=["fc"]).copy()
    # next-day simple return, which is what the position actually earns
    d["ret_next"] = d.groupby("symbol")["log_return"].shift(-1)
    d["ret_next"] = np.exp(d["ret_next"]) - 1.0
    return d.dropna(subset=["ret_next"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Exposure rules -- both expanding-window, no look-ahead
# ----------------------------------------------------------------------
def rule_vol_target(d: pd.DataFrame) -> pd.Series:
    target = d.groupby("symbol")["fc"].transform(
        lambda s: s.shift(1).expanding(MIN_HIST).median())
    return np.minimum(1.0, target / d["fc"]).rename("w")


def rule_threshold(d: pd.DataFrame) -> pd.Series:
    thr = d.groupby("symbol")["fc"].transform(
        lambda s: s.shift(1).expanding(MIN_HIST).quantile(0.80))
    return pd.Series(np.where(d["fc"] > thr, 0.5, 1.0), index=d.index,
                     name="w").where(thr.notna())


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def backtest(d: pd.DataFrame, w: pd.Series, cost_bp: float) -> pd.Series:
    """Daily portfolio excess return, net of costs. Equal weight across tickers."""
    n = d["symbol"].nunique()
    rf_d = RF_ANNUAL / 252.0
    x = d.assign(w=w).dropna(subset=["w"]).copy()
    x["excess"] = x["ret_next"] - rf_d            # cash earns rf, so excess = 0

    out = []
    for sym, g in x.groupby("symbol"):
        g = g.sort_values("date").copy()
        g["dw"] = g["w"].diff().abs().fillna(g["w"].iloc[0])
        g["contrib"] = (g["w"] * g["excess"] - g["dw"] * cost_bp / 10000.0) / n
        out.append(g[["date", "contrib"]])
    return pd.concat(out).groupby("date")["contrib"].sum().sort_index()


def benchmark(d: pd.DataFrame, cost_bp: float) -> pd.Series:
    n = d["symbol"].nunique()
    rf_d = RF_ANNUAL / 252.0
    out = []
    for sym, g in d.groupby("symbol"):
        g = g.sort_values("date").copy()
        c = np.zeros(len(g)); c[0] = cost_bp / 10000.0     # one initial purchase
        g["contrib"] = (g["ret_next"] - rf_d - c) / n
        out.append(g[["date", "contrib"]])
    return pd.concat(out).groupby("date")["contrib"].sum().sort_index()


def stats(r: pd.Series) -> dict:
    ann_r = r.mean() * 252
    ann_v = r.std(ddof=1) * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {"Ann. excess return": ann_r, "Ann. volatility": ann_v,
            "Sharpe": ann_r / ann_v if ann_v > 0 else np.nan,
            "Max drawdown": dd, "Days": len(r)}


# ----------------------------------------------------------------------
# Ledoit-Wolf robust Sharpe difference, circular block bootstrap
# ----------------------------------------------------------------------
def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return r.mean() / s * np.sqrt(252) if s > 0 else np.nan


def ledoit_wolf_test(a: np.ndarray, b: np.ndarray, block=BLOCK,
                     n_boot=N_BOOT, seed=SEED):
    """
    H0: Sharpe(b) - Sharpe(a) = 0, for two return series on the same dates.

    Ledoit and Wolf (2008) show that the usual Sharpe difference test is invalid
    when returns are non-normal and serially dependent, as daily equity returns
    are, and propose a bootstrap that preserves that dependence. A circular block
    bootstrap resamples contiguous blocks of days rather than individual days, so
    volatility clustering survives the resampling; the same block indices are
    applied to both series so the pairing is retained.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    obs = sharpe(b) - sharpe(a)
    nblocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nblocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        diffs[i] = sharpe(b[idx]) - sharpe(a[idx])
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": obs, "ci_low": lo, "ci_high": hi,
            "p": max(p, 1 / n_boot), "p_floored": p < 1 / n_boot}


def main() -> None:
    d = forecasts()
    hr("1. OUT-OF-SAMPLE WINDOW")
    print(f"  {d['date'].nunique()} trading days, {len(d):,} ticker-days, "
          f"{d['symbol'].nunique()} tickers")
    print(f"  {d['date'].min().date()} to {d['date'].max().date()}")
    print(f"  risk-free rate assumed {100*RF_ANNUAL:.0f}% annual; "
          f"returns are excess of it")

    rules = {"Volatility targeting (primary)": rule_vol_target(d),
             "Threshold de-risk 50% (robustness)": rule_threshold(d)}

    # Every series must cover identical dates or the Sharpe ratios are not
    # comparable. The expanding-window burn-in costs the strategies their first
    # MIN_HIST days, so the benchmark is restricted to the same window.
    common = None
    for w in rules.values():
        idx = backtest(d, w, 0).index
        common = idx if common is None else common.intersection(idx)
    print(f"  common evaluation window: {len(common)} trading days "
          f"(expanding-window burn-in costs the first "
          f"{d['date'].nunique() - len(common)})")

    rows, tests, series = [], [], {}
    for cost in COSTS_BP:
        bh = benchmark(d, cost).loc[common]
        s = stats(bh); s.update({"Strategy": "Buy-and-hold (benchmark)",
                                 "Cost (bp)": cost, "Turnover/yr": np.nan})
        rows.append(s); series[("bench", cost)] = bh

        for name, w in rules.items():
            r = backtest(d, w, cost).loc[common]
            bhc = bh
            n_t = d["symbol"].nunique()
            to = (d.assign(w=w).dropna(subset=["w"]).sort_values(["symbol", "date"])
                  .groupby("symbol")["w"].apply(lambda s: s.diff().abs().sum())
                  .sum() / n_t / len(common) * 252)
            st = stats(r); st.update({"Strategy": name, "Cost (bp)": cost,
                                      "Turnover/yr": to})
            rows.append(st); series[(name, cost)] = r

            lw = ledoit_wolf_test(bhc.values, r.values)
            tests.append({"Cost (bp)": cost, "Comparison": f"{name} vs buy-and-hold",
                          "Sharpe difference": round(lw["diff"], 4),
                          "95% CI low": round(lw["ci_low"], 4),
                          "95% CI high": round(lw["ci_high"], 4),
                          "p": ("< %.4f" % lw["p"]) if lw["p_floored"] else round(lw["p"], 4),
                          "Decision at .05": "reject H0" if lw["p"] < .05 else "retain H0"})

    perf = pd.DataFrame(rows)[["Strategy", "Cost (bp)", "Ann. excess return",
                               "Ann. volatility", "Sharpe", "Max drawdown",
                               "Turnover/yr", "Days"]]
    perf.round(4).to_csv(TAB_DIR / "rq4_performance.csv", index=False)
    hr("2. PERFORMANCE")
    print(perf.round(4).to_string(index=False))

    hr("3. LEDOIT-WOLF TEST (circular block bootstrap, block = %d days)" % BLOCK)
    tdf = pd.DataFrame(tests)
    tdf.to_csv(TAB_DIR / "rq4_tests.csv", index=False)
    print(tdf.to_string(index=False))

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    bh = series[("bench", 10)]
    ax[0].plot(bh.index, (1 + bh).cumprod(), color=GREY, lw=1.4, label="Buy-and-hold")
    for (name, col) in [("Volatility targeting (primary)", BLUE),
                        ("Threshold de-risk 50% (robustness)", RED)]:
        r = series[(name, 10)]
        ax[0].plot(r.index, (1 + r).cumprod(), color=col, lw=1.4,
                   label=name.split(" (")[0])
    ax[0].set(xlabel="Date", ylabel="Cumulative excess return (x)",
              title="(a) Growth of 1 unit, net of 10bp costs")
    ax[0].legend(frameon=False, fontsize=7.5)

    w = rule_vol_target(d)
    ex = d.assign(w=w).dropna(subset=["w"]).groupby("date")["w"].mean()
    ax[1].fill_between(ex.index, ex.values, 1.0, color=BLUE, alpha=0.30)
    ax[1].plot(ex.index, ex.values, color=BLUE, lw=1.0)
    ax[1].axhline(1.0, color=GREY, lw=1, ls=":")
    ax[1].set(xlabel="Date", ylabel="Average equity exposure", ylim=(0, 1.05),
              title="(b) Exposure under volatility targeting")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.savefig(FIG_DIR / "fig_rq4_backtest.png")
    plt.close(fig)

    hr("4. SUMMARY")
    for cost in COSTS_BP:
        b = perf[(perf.Strategy.str.startswith("Buy")) & (perf["Cost (bp)"] == cost)]
        p = perf[(perf.Strategy.str.startswith("Volatility")) & (perf["Cost (bp)"] == cost)]
        print(f"  at {cost:2d}bp: buy-and-hold Sharpe {b.Sharpe.iloc[0]:.3f}  "
              f"vs volatility targeting {p.Sharpe.iloc[0]:.3f}")
    print(f"\n  NOTE: {len(common)} evaluated trading days against the 1,558 required "
          f"by the power analysis;\n  a null result is inconclusive rather than "
          f"evidence of no effect.")
    print(f"\ntables -> {TAB_DIR.relative_to(C.ROOT)}   figures -> {FIG_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
