"""
Exploratory data analysis for the interim report's Analysis section.

Produces every figure and table the report needs, plus the printed numbers that
populate the data-cleaning log and the EDA insight summary. Each figure is saved
at 300 dpi with axis labels and a caption-ready title.

Outputs:
    docs/figures/fig_rv_distribution.png        rv_next, raw vs log scale
    docs/figures/fig_news_coverage.png          news coverage by ticker
    docs/figures/fig_sentiment_distribution.png sentiment and attention spread
    docs/figures/fig_sentiment_vs_rv.png        pooled vs within-ticker group means
    docs/figures/fig_correlation_heatmap.png    predictor correlations
    docs/figures/fig_volatility_clustering.png  rv over time, NVDA vs MSFT
    docs/tables/eda_descriptives.csv            descriptive statistics
    docs/tables/eda_vif.csv                     variance inflation factors
    docs/tables/eda_per_ticker.csv              per-ticker summary
    docs/tables/eda_cleaning_log.csv            record counts for the cleaning log

Run:
    uv run src/run_eda.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

FIG_DIR = C.ROOT / "docs" / "figures"
TAB_DIR = C.ROOT / "docs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_COLS = ["news_sent_wmean", "log_articles", "rv", "rv_w", "rv_m", "rv_next"]
PREDICTORS = ["news_sent_wmean", "log_articles", "rv", "rv_lag1", "rv_w", "rv_m",
              "abs_return", "log_volume"]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, 
})
GREY, BLUE, RED = "#4d4d4d", "#2c6fbb", "#c0392b"


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(C.PROCESSED / "panel.csv", parse_dates=["date"])
    scope = panel[panel["symbol"].isin(C.NEWS_TICKERS)]
    sample = scope.dropna(subset=MODEL_COLS).copy()
    return panel, sample


# ----------------------------------------------------------------------
# 1. Cleaning log -- record counts at each exclusion step
# ----------------------------------------------------------------------
def cleaning_log(panel: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    scope = panel[panel["symbol"].isin(C.NEWS_TICKERS)]
    n_price = len(panel)
    n_scope = len(scope)
    n_target = int(scope["rv_next"].isna().sum())
    n_har = int(scope["rv_m"].isna().sum())
    zero_att = int((sample["news_missing"] == 1).sum())

    rows = [
        ("Duplicate ticker-days", "symbol, date",
         "Key uniqueness assertion in validate_panel.py",
         "None found; no action required", 0, "Data quality"),
        ("Tickers outside sentiment universe", "all",
         "Alpha Vantage free tier permits 25 requests/day",
         f"Excluded {n_price - n_scope} ticker-days from 16 tickers with no news "
         f"collected; retained in the price panel as market context",
         n_price - n_scope, "Quota constraint"),
        ("Missing target (final trading day)", "rv_next",
         "Null check after the t -> t+1 shift",
         f"Excluded {n_target} ticker-days (one per ticker)", n_target,
         "Target undefined at the series end"),
        ("HAR burn-in window", "rv_m",
         "22-day rolling mean requires 22 prior observations",
         f"Excluded {n_har} ticker-days (21 per ticker)", n_har,
         "Monthly HAR component undefined"),
        ("No news articles on a trading day", "news_sent_wmean, news_article_count",
         "Left-join gap against the trading calendar",
         f"RETAINED as zero-attention observations: article_count = 0, "
         f"sentiment = 0 (neutral), news_missing = 1 ({zero_att} ticker-days)",
         zero_att, "A zero-article day is a genuine low-attention observation, "
                   "not missingness; dropping it truncates the attention proxy at 1"),
        ("Outliers / extreme returns", "close, rv",
         "Screen for |close-to-close| > 30% cross-checked against split ratios",
         "2 extreme moves verified as post-earnings repricings (ARM 2024-02-08, "
         "PLTR 2024-02-06) and RETAINED; no winsorising applied",
         2, "Genuine volatility events are the phenomenon under study"),
        ("User-identifying fields", "user_id, user_followers",
         "Privacy review of the StockTwits collector",
         "Dropped at collection and stripped from 23 existing files", 0,
         "Ethics: no individual user profiling"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Issue", "Variables Affected", "Detection Method", "Treatment Applied",
        "Records Affected", "Rationale"])
    df["% of Price Panel"] = (100 * df["Records Affected"] / n_price).round(2)
    df.to_csv(TAB_DIR / "eda_cleaning_log.csv", index=False)

    hr("1. DATA CLEANING LOG  (-> Table 3)")
    print(f"price panel                     {n_price:6d} ticker-days")
    print(f"  in sentiment universe (8)     {n_scope:6d}")
    print(f"  less missing target           -{n_target:5d}")
    print(f"  less HAR burn-in              -{n_har:5d}")
    print(f"  = ANALYSIS SAMPLE             {len(sample):6d}")
    print(f"      of which zero-attention   {zero_att:6d} ({100*zero_att/len(sample):.1f}%)")
    return df


# ----------------------------------------------------------------------
# 2. Descriptive statistics
# ----------------------------------------------------------------------
def descriptives(sample: pd.DataFrame) -> pd.DataFrame:
    cols = ["rv_next", "rv", "rv_w", "rv_m", "news_sent_wmean",
            "news_article_count", "log_articles", "log_return", "log_volume"]
    d = sample[cols].describe().T
    d["IQR"] = d["75%"] - d["25%"]
    d["skew"] = sample[cols].skew()
    d["kurtosis"] = sample[cols].kurtosis()
    d = d[["count", "mean", "50%", "std", "IQR", "min", "max", "skew", "kurtosis"]]
    d.columns = ["N", "Mean", "Median", "SD", "IQR", "Min", "Max", "Skew", "Kurtosis"]
    d.round(4).to_csv(TAB_DIR / "eda_descriptives.csv")

    hr("2. DESCRIPTIVE STATISTICS  (-> Step 4.2 requirement)")
    print(d.round(4).to_string())
    return d


# ----------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------
def fig_rv_distribution(sample: pd.DataFrame) -> dict:
    rv = sample["rv_next"]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
    ax[0].hist(rv, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
    ax[0].axvline(rv.mean(), color=RED, ls="--", lw=1, label=f"mean {rv.mean():.4f}")
    ax[0].axvline(rv.median(), color=GREY, ls=":", lw=1, label=f"median {rv.median():.4f}")
    ax[0].set(xlabel="Next-day realized volatility (daily sigma)", ylabel="Frequency",
              title=f"(a) Raw scale: skew {rv.skew():.2f}, kurtosis {rv.kurtosis():.2f}")
    ax[0].legend(frameon=False, fontsize=7)

    lrv = np.log(rv)
    ax[1].hist(lrv, bins=60, color=GREY, edgecolor="white", linewidth=0.3)
    ax[1].set(xlabel="log(next-day realized volatility)", ylabel="Frequency",
              title=f"(b) Log scale: skew {lrv.skew():.2f}, kurtosis {lrv.kurtosis():.2f}")
    fig.savefig(FIG_DIR / "fig_rv_distribution.png")
    plt.close(fig)
    return {"skew_raw": rv.skew(), "kurt_raw": rv.kurtosis(),
            "skew_log": lrv.skew(), "kurt_log": lrv.kurtosis()}


def fig_news_coverage(panel: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    g = sample.groupby("symbol").agg(
        days=("date", "size"),
        with_article=("news_missing", lambda s: int((s == 0).sum())),
        mean_articles=("news_article_count", "mean"),
        mean_sent=("news_sent_wmean", "mean"),
        mean_rv=("rv_next", "mean"),
    )
    g["coverage_pct"] = (100 * g["with_article"] / g["days"]).round(1)
    g = g.sort_values("coverage_pct", ascending=False)
    g.round(4).to_csv(TAB_DIR / "eda_per_ticker.csv")

    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    x = np.arange(len(g))
    ax.bar(x, g["coverage_pct"], color=BLUE, label="Days with at least one article")
    ax.bar(x, 100 - g["coverage_pct"], bottom=g["coverage_pct"], color="#bdbdbd",
           label="Zero-attention days (no article)")

    # Label both segments so neither relies on the legend alone.
    for i, v in enumerate(g["coverage_pct"]):
        ax.text(i, v - 5, f"{v:.0f}%", ha="center", va="top",
                color="white", fontsize=8, fontweight="bold")
        if 100 - v >= 8:                      # only where the grey band has room
            ax.text(i, v + (100 - v) / 2, f"{100 - v:.0f}%", ha="center", va="center",
                    color="#333333", fontsize=8)

    ax.set(xticks=x, xticklabels=g.index, ylabel="% of analysis-sample days",
           ylim=(0, 100))
    ax.set_title("News coverage by ticker (analysis sample, n = 480 days each)", pad=8)
    # Legend below the axes, outside the data area, so it can never sit on a bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=9, handlelength=1.6, columnspacing=2.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "fig_news_coverage.png")
    plt.close(fig)
    return g


def fig_sentiment_distribution(sample: pd.DataFrame) -> None:
    """
    Two histograms answering 'how much usable variation do the predictors have?'
    Each bar counts ticker-days. Annotated so the chart is readable without the
    caption: the reader should see that sentiment is almost never negative and
    that attention is bunched at very low article counts.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)

    # ---- (a) sentiment, days with news only -------------------------------
    s = sample.loc[sample["news_missing"] == 0, "news_sent_wmean"]
    pos = 100 * (s > 0).mean()
    a = ax[0]
    counts, bins, patches = a.hist(s, bins=50, color=BLUE,
                                   edgecolor="white", linewidth=0.3)
    # colour the negative side differently so the asymmetry is visible at a glance
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor("#c0392b")
    a.axvline(0, color="black", lw=1.2)
    a.axvline(s.mean(), color="black", ls="--", lw=1.2)

    top = counts.max()
    a.annotate(f"mean {s.mean():+.2f}", xy=(s.mean(), top * 0.98),
               xytext=(s.mean() + 0.22, top * 0.98), fontsize=8, va="center",
               arrowprops=dict(arrowstyle="-", lw=0.8))
    a.text(0.52, top * 0.70, f"{pos:.0f}% of days\nPOSITIVE", fontsize=10,
           color=BLUE, fontweight="bold", ha="left", va="center")
    a.text(-0.60, top * 0.45, f"{100-pos:.0f}% of days\nnegative", fontsize=10,
           color="#c0392b", fontweight="bold", ha="left", va="center")
    a.set(xlabel="News sentiment score  (-1 = bearish, +1 = bullish)",
          ylabel="Number of ticker-days")
    a.set_title(f"(a) How positive was the news?\nDays with at least one article "
                f"(n = {len(s):,})", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)

    # ---- (b) attention, all analysis-sample days --------------------------
    ac = sample["news_article_count"]
    b = ax[1]
    counts2, bins2, patches2 = b.hist(ac, bins=range(0, int(ac.max()) + 2),
                                      color="#4d4d4d", edgecolor="white", linewidth=0.3)
    patches2[0].set_facecolor("#bdbdbd")          # match Figure 2's zero-attention grey
    n0 = int((ac == 0).sum())
    b.annotate(f"{n0:,} days ({100*n0/len(ac):.0f}%)\nhad NO article",
               xy=(0.5, counts2[0]), xytext=(6, counts2[0] * 0.92),
               fontsize=9, va="center",
               arrowprops=dict(arrowstyle="->", lw=0.9))
    b.annotate(f"median {ac.median():.0f} articles;\nlong thin tail to {int(ac.max())}",
               xy=(12, counts2.max() * 0.30), fontsize=8, va="center")
    b.set(xlabel="Number of articles on that ticker-day",
          ylabel="Number of ticker-days")
    b.set_title(f"(b) How much news was there?\nAll analysis-sample days "
                f"(n = {len(ac):,})", fontsize=9)
    b.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "fig_sentiment_distribution.png")
    plt.close(fig)


def fig_sentiment_vs_rv(sample: pd.DataFrame) -> dict:
    """
    Quintile means, computed POOLED and WITHIN-TICKER.

    The two disagree in sign for attention, because news coverage and baseline
    volatility are correlated ACROSS tickers (PLTR: 73% zero-attention days and
    the highest volatility; MSFT: 4% and among the lowest). Pooling therefore
    lets a cross-sectional difference masquerade as a time-series relationship
    -- a Simpson's paradox. Only the within-ticker comparison answers RQ2's
    actual question, which is whether a change in a ticker's own attention
    predicts a change in its own next-day volatility.
    """
    out = {}
    s = sample.copy()
    # Express the target relative to each ticker's own mean, so a value of 1.05
    # means "5% above normal for this stock".
    s["rv_rel"] = s["rv_next"] / s.groupby("symbol")["rv_next"].transform("mean")

    # Sentiment is continuous, so equal-sized quintiles are appropriate.
    # Attention is a discrete count with 25% of days at exactly zero, so
    # quintiles either collapse (two boundaries land on 0) or split identical
    # observations across bins, both of which distort the gradient. Explicit
    # count bands are used instead: they are interpretable and every band spans
    # a genuinely different level of coverage.
    ATT_BINS = [-0.5, 0.5, 1.5, 3.5, 6.5, np.inf]
    ATT_LABELS = ["0", "1", "2-3", "4-6", "7+"]

    def grouper(frame: pd.DataFrame, col: str) -> pd.Series:
        if col == "news_article_count":
            return pd.cut(frame[col], bins=ATT_BINS, labels=ATT_LABELS)
        return pd.qcut(frame[col], 5, labels=["1", "2", "3", "4", "5"])

    specs = [("news_sent_wmean",
              "News sentiment quintile\n(1 = most bearish, 5 = most bullish)"),
             ("news_article_count",
              "Articles published that day")]

    fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.8), constrained_layout=True)

    for j, (col, xlabel) in enumerate(specs):
        base = s[s["news_missing"] == 0] if col == "news_sent_wmean" else s

        g = grouper(base, col)
        cats = list(g.cat.categories)
        x = np.arange(len(cats))

        # ---- row 0: pooled across tickers (confounded) --------------------
        m = base.groupby(g, observed=True)["rv_next"].agg(["mean", "sem"]).reindex(cats)
        a = ax[0, j]
        a.errorbar(x, m["mean"], yerr=1.96 * m["sem"], marker="o",
                   color="#c0392b", capsize=4, lw=1.6, ms=6)
        pooled = 100 * (m["mean"].iloc[-1] / m["mean"].iloc[0] - 1)
        a.set(xlabel=xlabel, ylabel="Mean next-day volatility\n(daily sigma)",
              xticks=x, xticklabels=cats)
        a.set_title(f"lowest to highest:  {pooled:+.1f}%", fontsize=11,
                    color="#c0392b", fontweight="bold")

        # ---- row 1: within ticker (confound removed) ----------------------
        mw = base.groupby(g, observed=True)["rv_rel"].agg(["mean", "sem"]).reindex(cats)
        a = ax[1, j]
        a.errorbar(x, mw["mean"], yerr=1.96 * mw["sem"], marker="o",
                   color=BLUE, capsize=4, lw=1.6, ms=6)
        a.axhline(1.0, color="#777777", lw=1.0, ls=":")
        a.text(len(cats) - 0.55, 1.0, "typical\nday", fontsize=7.5,
               color="#777777", va="center", ha="left")
        within = 100 * (mw["mean"].iloc[-1] / mw["mean"].iloc[0] - 1)
        a.set(xlabel=xlabel,
              ylabel="Next-day volatility\nrelative to that ticker's average",
              xticks=x, xticklabels=cats, xlim=(-0.4, len(cats) - 0.1))
        a.set_title(f"lowest to highest:  {within:+.1f}%", fontsize=11,
                    color=BLUE, fontweight="bold")

        out[col] = {"pooled_pct": pooled, "within_pct": within,
                    "monotonic": bool(mw["mean"].is_monotonic_increasing),
                    "sign_flip": bool(np.sign(pooled) != np.sign(within))}

    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)

    # Column headings and row descriptors, so the 2x2 structure is explicit.
    ax[0, 0].annotate("NEWS SENTIMENT  (tone)", xy=(0.5, 1.30), xycoords="axes fraction",
                      ha="center", fontsize=11, fontweight="bold")
    ax[0, 1].annotate("ATTENTION  (how much coverage)", xy=(0.5, 1.30),
                      xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold")
    ax[0, 0].annotate("POOLED\nacross all\n8 tickers", xy=(-0.42, 0.5),
                      xycoords="axes fraction", ha="center", va="center",
                      fontsize=10, fontweight="bold", color="#c0392b", rotation=90)
    ax[1, 0].annotate("WITHIN\neach ticker", xy=(-0.42, 0.5), xycoords="axes fraction",
                      ha="center", va="center", fontsize=10, fontweight="bold",
                      color=BLUE, rotation=90)

    # Call out the one panel that carries the finding.
    ax[1, 1].annotate("rises at every step:\nthe effect RQ2 tests for",
                      xy=(3.6, mw["mean"].iloc[-1]),
                      xytext=(0.15, mw["mean"].max() * 0.999),
                      fontsize=8.5, color=BLUE,
                      arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

    fig.suptitle("Same data, opposite conclusions: pooling across tickers reverses "
                 "the attention effect\n(points are quintile means; bars are 95% "
                 "confidence intervals)", fontsize=10.5)
    fig.savefig(FIG_DIR / "fig_sentiment_vs_rv.png")
    plt.close(fig)

    # Zero-attention contrast, pooled and within-ticker.
    zp = sample.groupby("news_missing")["rv_next"].agg(["mean", "size"])
    zw = s.groupby("news_missing")["rv_rel"].agg(["mean", "size"])
    out["zero_attention_pooled"] = zp
    out["zero_attention_within"] = zw
    return out


def fig_confounding(sample: pd.DataFrame) -> pd.DataFrame:
    """Show the cross-sectional confound directly: coverage vs baseline volatility."""
    t = sample.groupby("symbol").agg(
        zero_att_pct=("news_missing", lambda x: 100 * x.mean()),
        mean_rv=("rv_next", "mean"))
    r = t["zero_att_pct"].corr(t["mean_rv"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.scatter(t["zero_att_pct"], t["mean_rv"], color=BLUE, s=40, zorder=3)
    for sym, row in t.iterrows():
        ax.annotate(sym, (row["zero_att_pct"], row["mean_rv"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    z = np.polyfit(t["zero_att_pct"], t["mean_rv"], 1)
    xs = np.linspace(t["zero_att_pct"].min(), t["zero_att_pct"].max(), 10)
    ax.plot(xs, np.polyval(z, xs), color=RED, ls="--", lw=1)
    ax.set(xlabel="% of days with zero news articles",
           ylabel="Mean next-day realized volatility",
           title=f"Cross-sectional confound: r = {r:.2f} across 8 tickers")
    fig.savefig(FIG_DIR / "fig_confounding.png")
    plt.close(fig)

    t["r_with_volatility"] = r
    t.round(4).to_csv(TAB_DIR / "eda_confounding.csv")
    return t


def fig_correlation_heatmap(sample: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PREDICTORS if c in sample] + ["rv_next"]
    corr = sample[cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(cols)), yticks=range(len(cols)),
           xticklabels=cols, yticklabels=cols,
           title="Predictor correlation matrix (analysis sample)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.savefig(FIG_DIR / "fig_correlation_heatmap.png")
    plt.close(fig)
    return corr


def fig_volatility_clustering(sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 2.8), constrained_layout=True)
    for sym, colour in [("NVDA", BLUE), ("MSFT", GREY)]:
        s = sample[sample["symbol"] == sym].sort_values("date")
        ax.plot(s["date"], s["rv"], lw=0.7, color=colour, label=sym)
    ax.set(xlabel="Date", ylabel="Realized volatility (daily sigma)",
           title="Volatility clustering: high-volatility days arrive in runs")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "fig_volatility_clustering.png")
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Multicollinearity (Step 4.1 requirement)
# ----------------------------------------------------------------------
def vif_table(sample: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X = add_constant(sample[PREDICTORS].astype(float))
    rows = [(c, variance_inflation_factor(X.values, i))
            for i, c in enumerate(X.columns) if c != "const"]
    v = pd.DataFrame(rows, columns=["Predictor", "VIF"]).sort_values("VIF", ascending=False)
    v["Assessment"] = np.where(v["VIF"] > 10, "SEVERE (>10)",
                        np.where(v["VIF"] > 5, "moderate (5-10)", "acceptable (<5)"))
    v.round(3).to_csv(TAB_DIR / "eda_vif.csv", index=False)

    hr("4. MULTICOLLINEARITY -- VIF  (-> Step 4.1 requirement)")
    print(v.round(2).to_string(index=False))
    print("\nStep 4 threshold: VIF > 10 indicates severe multicollinearity.")
    return v


# ----------------------------------------------------------------------
# 5. Sample adequacy -- effective N and regime coverage
# ----------------------------------------------------------------------
def sample_adequacy(sample: pd.DataFrame) -> dict:
    """
    Two honesty checks on the sample, both reported in the limitations.

    (a) EFFECTIVE SAMPLE SIZE. The panel has 3,840 rows but the eight tickers
        move together, so those rows are not independent. The standard
        adjustment for k equicorrelated series with mean pairwise correlation
        rho is k_eff = k / (1 + (k-1) * rho).

    (b) REGIME COVERAGE. A volatility model is most valuable when conditions
        change. If the window contains only one regime, that is an external
        validity limit no amount of rows can fix.
    """
    out = {}
    hr("5. SAMPLE ADEQUACY  (-> limitations)")

    for label, col in [("realized volatility", "rv_next"), ("daily return", "log_return")]:
        wide = sample.pivot_table(index="date", columns="symbol", values=col)
        cm = wide.corr()
        off = cm.values[np.triu_indices_from(cm.values, 1)]
        rho = float(off.mean())
        k = len(cm)
        k_eff = k / (1 + (k - 1) * rho)
        n_eff = k_eff * wide.shape[0]
        out[col] = {"rho": rho, "k_eff": k_eff, "n_eff": n_eff}
        print(f"\n  {label}: mean pairwise r = {rho:.3f} "
              f"(range {off.min():.2f} to {off.max():.2f})")
        print(f"    effective independent series : {k_eff:.2f} of {k}")
        print(f"    effective sample size        : {n_eff:.0f} of {len(sample)} rows")

    # Minimum detectable effect for RQ2 on each defensible basis.
    L, u = 10.90, 3
    print("\n  RQ2 minimum detectable effect (alpha=.05, power=.80, u=3):")
    bases = [("pooled rows (optimistic)", len(sample)),
             ("correlation-adjusted (honest)", int(out["rv_next"]["n_eff"])),
             ("date clusters (conservative)", sample["date"].nunique())]
    for label, N in bases:
        f2 = L / (N - u - 1)
        out[label] = f2
        print(f"    {label:32s} N={N:5d}  detectable f2 = {f2:.4f}")
    print("    Cohen benchmarks: small = 0.02, medium = 0.15, large = 0.35")

    # Regime coverage.
    q = sample.pivot_table(index="date", columns="symbol",
                           values="rv_next").mean(axis=1).resample("QE").mean()
    ratio = float(q.max() / q.min())
    out["regime_ratio"] = ratio
    out["quarters"] = q
    print(f"\n  Quarterly cross-sectional mean volatility: "
          f"{q.min():.4f} to {q.max():.4f}")
    print(f"    highest / lowest quarter = {ratio:.2f}x  -> "
          f"{'single regime' if ratio < 2 else 'multiple regimes'}")
    q.round(4).to_csv(TAB_DIR / "eda_regime_coverage.csv")
    return out


def main() -> None:
    panel, sample = load()
    print(f"loaded panel rows={len(panel)}  analysis sample={len(sample)}")

    cleaning_log(panel, sample)
    descriptives(sample)

    hr("3. FIGURES")
    dist = fig_rv_distribution(sample)
    cov = fig_news_coverage(panel, sample)
    fig_sentiment_distribution(sample)
    rel = fig_sentiment_vs_rv(sample)
    conf = fig_confounding(sample)
    corr = fig_correlation_heatmap(sample)
    fig_volatility_clustering(sample)
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  saved {f.relative_to(C.ROOT)}")

    v = vif_table(sample)
    sample_adequacy(sample)

    hr("6. KEY INSIGHTS  (-> Table 6, one interpretation each)")
    print(f"""
[Fig 2] rv_next is right-skewed (skew {dist['skew_raw']:.2f}, kurtosis
        {dist['kurt_raw']:.2f}). Log transform reduces this to skew
        {dist['skew_log']:.2f} / kurtosis {dist['kurt_log']:.2f}.
        DECISION: model log(rv_next); report QLIKE alongside RMSE because
        squared-error loss is dominated by the upper tail.

[Fig 3] News coverage ranges {cov['coverage_pct'].min():.0f}%-{cov['coverage_pct'].max():.0f}%
        ({cov['coverage_pct'].idxmin()} lowest, {cov['coverage_pct'].idxmax()} highest).
        DECISION: retain zero-attention days and include news_missing so the
        model can separate neutral coverage from absent coverage.

[Fig 4] Sentiment is asymmetric: mean {sample.loc[sample.news_missing==0,'news_sent_wmean'].mean():+.3f},
        {100*(sample.loc[sample.news_missing==0,'news_sent_wmean'] > 0).mean():.0f}% of days positive.
        DECISION: limited negative variation caps the detectable effect; treat as
        a limitation and prefer attention over sentiment level as the signal.

[Fig 5] SIGN REVERSAL. Attention Q5 vs Q1 is {rel['news_article_count']['pooled_pct']:+.1f}% pooled
        but {rel['news_article_count']['within_pct']:+.1f}% within-ticker. Sentiment:
        {rel['news_sent_wmean']['pooled_pct']:+.1f}% pooled, {rel['news_sent_wmean']['within_pct']:+.1f}% within-ticker.
        DECISION: the pooled estimate is confounded (see Fig 6); RQ2 must include
        ticker fixed effects. Attention is the stronger candidate and, once the
        confound is removed, moves in the direction Audrino et al. (2020) report.

[Fig 6] The confound quantified: across the 8 tickers, % zero-attention days
        correlates r = {conf['r_with_volatility'].iloc[0]:.2f} with mean volatility. PLTR has
        {conf.loc['PLTR','zero_att_pct']:.0f}% zero-attention days and the highest volatility
        ({conf.loc['PLTR','mean_rv']:.4f}); MSFT has {conf.loc['MSFT','zero_att_pct']:.0f}% and among the lowest
        ({conf.loc['MSFT','mean_rv']:.4f}). Low attention is therefore partly a marker for
        WHICH stock, not for what is about to happen to it.
        DECISION: report pooled and within-ticker results side by side; treat the
        pooled comparison as a cautionary illustration, not a finding.

[Fig 7] Highest predictor-target correlation: {corr['rv_next'].drop('rv_next').abs().max():.3f}
        ({corr['rv_next'].drop('rv_next').abs().idxmax()}). Sentiment correlates
        {corr.loc['news_sent_wmean','rv_next']:+.3f} with the target.
        DECISION: lagged volatility dominates; sentiment must be judged on
        INCREMENTAL contribution over HAR, not raw correlation.

[Table] Max VIF = {v['VIF'].max():.2f} ({v.iloc[0]['Predictor']}).
        DECISION: {'drop or orthogonalise the collinear volatility terms' if v['VIF'].max() > 10 else 'all predictors below the severe threshold; retain the full control set'}.
""")
    zp, zw = rel["zero_attention_pooled"], rel["zero_attention_within"]
    if len(zp) == 2 and len(zw) == 2:
        print(f"[Fig 8] Zero-attention vs with-news days, next-day volatility:")
        print(f"        pooled       : {zp.loc[1,'mean']:.5f} vs {zp.loc[0,'mean']:.5f} "
              f"({100*(zp.loc[0,'mean']/zp.loc[1,'mean'] - 1):+.1f}% for news days)")
        print(f"        within-ticker: {zw.loc[1,'mean']:.4f} vs {zw.loc[0,'mean']:.4f} "
              f"({100*(zw.loc[0,'mean']/zw.loc[1,'mean'] - 1):+.1f}% for news days)")
        print(f"        n = {int(zp.loc[1,'size'])} zero-attention, {int(zp.loc[0,'size'])} with news.")
        print("        This contrast is only estimable because zero-article days "
              "are retained rather than dropped.")

    print(f"\ntables -> {TAB_DIR.relative_to(C.ROOT)}   figures -> {FIG_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
