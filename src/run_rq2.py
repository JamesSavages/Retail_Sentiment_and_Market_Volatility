"""
RQ2: Does day-t sentiment predict next-day realized volatility, controlling for
lagged volatility and volume?

    H0: the sentiment coefficients are jointly zero
    H1: at least one is non-zero
    Tests: incremental F-test (in-sample) and Diebold-Mariano (out-of-sample)

SPECIFICATION
    Baseline   log(rv_next) ~ rv + rv_w + rv_m + log_volume + ticker FE
    Augmented  baseline + news_sent_wmean + log_articles + news_missing

    Ticker fixed effects are NOT optional here. The exploratory analysis showed
    that pooling without them reverses the sign of the attention effect, because
    coverage and baseline volatility are correlated across firms (see Figure 5 in
    the report). Fixed effects restrict identification to within-ticker variation,
    which is what the research question actually asks about.

    The target is logged because raw realized volatility is right-skewed
    (skew 1.81); QLIKE is computed on the variance scale, following Patton (2011).

VALIDATION
    Expanding-window walk-forward, split on DATES so that all eight tickers move
    through the windows together. k-fold is not used: random folds would place
    future days in the training set.

Outputs:
    docs/tables/rq2_performance.csv    fold-by-fold and pooled out-of-sample metrics
    docs/tables/rq2_coefficients.csv   full-sample coefficients with clustered SEs
    docs/tables/rq2_tests.csv          incremental F-test and Diebold-Mariano
    docs/figures/fig_rq2_forecast.png  predicted vs actual, and error by fold

Run:
    uv run src/run_rq2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

FIG_DIR = C.ROOT / "docs" / "figures"
TAB_DIR = C.ROOT / "docs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HAR = ["rv", "rv_w", "rv_m", "log_volume"]
SENT = ["news_sent_wmean", "log_articles", "news_missing"]
N_FOLDS = 6
MIN_TRAIN_FRAC = 0.40
BLUE, GREY, RED = "#2c6fbb", "#4d4d4d", "#c0392b"

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def load() -> pd.DataFrame:
    p = pd.read_csv(C.PROCESSED / "panel.csv", parse_dates=["date"])
    need = HAR + SENT + ["rv_next"]
    d = p[p["symbol"].isin(C.NEWS_TICKERS)].dropna(subset=need).copy()
    d["y"] = np.log(d["rv_next"])
    d = pd.concat([d, pd.get_dummies(d["symbol"], prefix="fe", drop_first=True,
                                     dtype=float)], axis=1)
    return d.sort_values(["date", "symbol"]).reset_index(drop=True)


def design(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    fe = [c for c in d.columns if c.startswith("fe_")]
    return sm.add_constant(d[cols + fe].astype(float), has_constant="add")


# ----------------------------------------------------------------------
# Losses. QLIKE is defined on VARIANCE, so the logged forecast is exponentiated
# back to a volatility and then squared (Patton, 2011).
# ----------------------------------------------------------------------
def rmse(y_log, p_log):
    return float(np.sqrt(np.mean((np.exp(y_log) - np.exp(p_log)) ** 2)))


def qlike_losses(y_log, p_log):
    a = np.exp(y_log) ** 2
    h = np.exp(p_log) ** 2
    r = a / h
    return r - np.log(r) - 1.0


def diebold_mariano(l1, l2, h: int = 1):
    """Test equal predictive accuracy of two loss series. Returns (DM, p)."""
    from scipy.stats import norm
    d = np.asarray(l1) - np.asarray(l2)
    n = len(d)
    dbar = d.mean()
    gam = [np.cov(d[:-k], d[k:], bias=True)[0, 1] if k else d.var() for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    return float(dm), float(2 * norm.sf(abs(dm)))


def walk_forward(d: pd.DataFrame):
    """Expanding-window folds, split on dates so tickers advance together."""
    dates = np.sort(d["date"].unique())
    start = int(len(dates) * MIN_TRAIN_FRAC)
    edges = np.linspace(start, len(dates), N_FOLDS + 1).astype(int)
    for i in range(N_FOLDS):
        tr = d["date"] < dates[edges[i]]
        te = (d["date"] >= dates[edges[i]]) & (
            d["date"] < dates[edges[i + 1]] if edges[i + 1] < len(dates) else True)
        if tr.sum() and te.sum():
            yield i + 1, d[tr], d[te]


def main() -> None:
    d = load()
    hr("1. SAMPLE")
    print(f"  ticker-days {len(d):,} | tickers {d['symbol'].nunique()} | "
          f"dates {d['date'].nunique()} | {d['date'].min().date()} to {d['date'].max().date()}")
    print(f"  target: log(rv_next); baseline {HAR} + ticker fixed effects")
    print(f"  sentiment block: {SENT}")

    # ---- full-sample fit: coefficients and the incremental F-test ----------
    hr("2. FULL-SAMPLE FIT (in-sample inference)")
    Xb, Xa, y = design(d, HAR), design(d, HAR + SENT), d["y"]
    groups = d["date"]
    mb = sm.OLS(y, Xb).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ma = sm.OLS(y, Xa).fit(cov_type="cluster", cov_kwds={"groups": groups})

    print(f"  baseline  R2 {mb.rsquared:.4f}  adj {mb.rsquared_adj:.4f}  "
          f"AIC {mb.aic:.1f}  BIC {mb.bic:.1f}")
    print(f"  augmented R2 {ma.rsquared:.4f}  adj {ma.rsquared_adj:.4f}  "
          f"AIC {ma.aic:.1f}  BIC {ma.bic:.1f}")
    print(f"  incremental R2: {ma.rsquared - mb.rsquared:+.5f}")

    ftest = ma.f_test(" = 0, ".join(SENT) + " = 0")
    fstat, fp = float(np.squeeze(ftest.statistic)), float(np.squeeze(ftest.pvalue))
    print(f"\n  Incremental F-test on {SENT}")
    print(f"    F = {fstat:.3f}   p = {fp:.4g}   -> "
          f"{'reject H0' if fp < .05 else 'retain H0'} at the 5% level")

    coef = pd.DataFrame({"coef": ma.params, "std_err": ma.bse,
                         "t": ma.tvalues, "p": ma.pvalues})
    coef.loc[[c for c in SENT + HAR]].round(5).to_csv(TAB_DIR / "rq2_coefficients.csv")
    print("\n  sentiment coefficients (SEs clustered by date):")
    print(coef.loc[SENT].round(4).to_string())

    # ---- walk-forward out-of-sample --------------------------------------
    hr("3. WALK-FORWARD OUT-OF-SAMPLE")
    rows, lb_all, la_all, lg_all = [], [], [], []
    for k, tr, te in walk_forward(d):
        Xtr_b, Xte_b = design(tr, HAR), design(te, HAR)
        Xtr_a, Xte_a = design(tr, HAR + SENT), design(te, HAR + SENT)
        fb = sm.OLS(tr["y"], Xtr_b).fit()
        fa = sm.OLS(tr["y"], Xtr_a).fit()
        pb = fb.predict(Xte_b[Xtr_b.columns])
        pa = fa.predict(Xte_a[Xtr_a.columns])

        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          max_depth=4, random_state=42)
        g.fit(tr[HAR + SENT], tr["y"])
        pg = g.predict(te[HAR + SENT])

        lb, la, lg = (qlike_losses(te["y"], p) for p in (pb, pa, pg))
        lb_all += list(lb); la_all += list(la); lg_all += list(lg)
        rows.append({"Fold": k, "Train n": len(tr), "Test n": len(te),
                     "Test from": te["date"].min().date(),
                     "RMSE base": rmse(te["y"], pb), "RMSE aug": rmse(te["y"], pa),
                     "RMSE GBM": rmse(te["y"], pg),
                     "QLIKE base": lb.mean(), "QLIKE aug": la.mean(),
                     "QLIKE GBM": lg.mean()})
        print(f"  fold {k}: train {len(tr):5d}  test {len(te):4d} from "
              f"{te['date'].min().date()}  QLIKE base {lb.mean():.4f} -> "
              f"aug {la.mean():.4f}  GBM {lg.mean():.4f}")

    perf = pd.DataFrame(rows)
    pooled = {"Fold": "POOLED", "Train n": "", "Test n": perf["Test n"].sum(),
              "Test from": "", "RMSE base": np.nan, "RMSE aug": np.nan,
              "RMSE GBM": np.nan, "QLIKE base": np.mean(lb_all),
              "QLIKE aug": np.mean(la_all), "QLIKE GBM": np.mean(lg_all)}
    perf = pd.concat([perf, pd.DataFrame([pooled])], ignore_index=True)
    perf.round(5).to_csv(TAB_DIR / "rq2_performance.csv", index=False)

    # ---- Diebold-Mariano --------------------------------------------------
    hr("4. DIEBOLD-MARIANO (out-of-sample, QLIKE losses)")
    tests = []
    for nm, l2 in [("HAR + sentiment", la_all), ("Gradient boosting", lg_all)]:
        dm, p = diebold_mariano(lb_all, l2)
        better = nm if np.mean(l2) < np.mean(lb_all) else "HAR baseline"
        tests.append({"Comparison": f"HAR baseline vs {nm}", "DM": round(dm, 3),
                      "p": p, "Lower QLIKE": better,
                      "Decision at .05": "reject equal accuracy" if p < .05
                                         else "retain equal accuracy"})
        print(f"  vs {nm:20s} DM = {dm:+.3f}  p = {p:.4g}  "
              f"-> lower loss: {better}")

    tests.append({"Comparison": "Incremental F-test on sentiment block",
                  "DM": round(fstat, 3), "p": fp, "Lower QLIKE": "",
                  "Decision at .05": "reject H0" if fp < .05 else "retain H0"})
    pd.DataFrame(tests).to_csv(TAB_DIR / "rq2_tests.csv", index=False)

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    f = perf[perf["Fold"] != "POOLED"]
    x = np.arange(len(f))
    w = 0.27
    ax[0].bar(x - w, f["QLIKE base"], w, color=GREY, label="HAR baseline")
    ax[0].bar(x, f["QLIKE aug"], w, color=BLUE, label="HAR + sentiment")
    ax[0].bar(x + w, f["QLIKE GBM"], w, color=RED, label="Gradient boosting")
    ax[0].set(xlabel="Walk-forward fold", ylabel="QLIKE (lower is better)",
              xticks=x, xticklabels=f["Fold"], title="(a) Out-of-sample loss by fold")
    ax[0].legend(frameon=False, fontsize=7.5)

    diff = 100 * (f["QLIKE aug"].values / f["QLIKE base"].values - 1)
    ax[1].bar(x, diff, color=np.where(diff < 0, BLUE, RED))
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set(xlabel="Walk-forward fold",
              ylabel="QLIKE change vs baseline (%)", xticks=x,
              xticklabels=f["Fold"],
              title="(b) Does sentiment help?  (below 0 = yes)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "fig_rq2_forecast.png")
    plt.close(fig)

    hr("5. SUMMARY")
    imp = 100 * (np.mean(la_all) / np.mean(lb_all) - 1)
    print(f"  pooled out-of-sample QLIKE  baseline {np.mean(lb_all):.5f}  "
          f"augmented {np.mean(la_all):.5f}  ({imp:+.2f}%)")
    print(f"  incremental in-sample R2    {ma.rsquared - mb.rsquared:+.5f} "
          f"(F = {fstat:.2f}, p = {fp:.4g})")
    print(f"\ntables -> {TAB_DIR.relative_to(C.ROOT)}   "
          f"figures -> {FIG_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
