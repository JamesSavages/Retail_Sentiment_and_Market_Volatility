"""
RQ3: Do sentiment features improve next-day return-direction prediction beyond a
technicals-only baseline?

    H0: accuracy(augmented) <= accuracy(baseline)
    H1: accuracy(augmented) >  accuracy(baseline)
    Test: McNemar's test (paired -- both models score identical test days)

WHAT TO EXPECT
    Daily return direction is close to a coin flip and the literature reports
    that sentiment predicts volatility far more reliably than returns, which are
    nearer efficient. A null result here is an anticipated and reportable
    outcome, not a failure: it is the direct counterpart to RQ2's positive
    finding and supports the study's decision to target volatility.

DESIGN
    Identical panel, ticker fixed effects, and walk-forward folds as RQ2, so the
    only difference from that question is the target and the model class.
    Features are standardised for logistic regression (scale-sensitive) and left
    raw for gradient boosting (invariant to monotonic rescaling). The scaler is
    fitted on each training window only.

Outputs:
    docs/tables/rq3_performance.csv   fold-by-fold and pooled out-of-sample metrics
    docs/tables/rq3_tests.csv         McNemar comparisons
    docs/figures/fig_rq3_direction.png  accuracy by fold and ROC

Run:
    uv run src/run_rq3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, roc_curve)  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

FIG_DIR = C.ROOT / "docs" / "figures"
TAB_DIR = C.ROOT / "docs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TECH = ["rv", "rv_w", "rv_m", "log_volume", "log_return", "abs_return"]
SENT = ["news_sent_wmean", "log_articles", "news_missing"]
N_FOLDS = 6
MIN_TRAIN_FRAC = 0.40
BLUE, GREY, RED = "#2c6fbb", "#4d4d4d", "#c0392b"

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def load() -> pd.DataFrame:
    p = pd.read_csv(C.PROCESSED / "panel.csv", parse_dates=["date"])
    d = p[p["symbol"].isin(C.NEWS_TICKERS)].sort_values(["symbol", "date"]).copy()
    # Target: direction of the NEXT day's return, within each ticker.
    d["ret_next"] = d.groupby("symbol")["log_return"].shift(-1)
    d["y"] = (d["ret_next"] > 0).astype(int)
    d = d.dropna(subset=TECH + SENT + ["ret_next"])
    fe = pd.get_dummies(d["symbol"], prefix="fe", drop_first=True, dtype=float)
    d = pd.concat([d, fe], axis=1)
    return d.sort_values(["date", "symbol"]).reset_index(drop=True)


def walk_forward(d: pd.DataFrame):
    dates = np.sort(d["date"].unique())
    start = int(len(dates) * MIN_TRAIN_FRAC)
    edges = np.linspace(start, len(dates), N_FOLDS + 1).astype(int)
    for i in range(N_FOLDS):
        tr = d["date"] < dates[edges[i]]
        te = ((d["date"] >= dates[edges[i]]) &
              (d["date"] < dates[edges[i + 1]] if edges[i + 1] < len(dates) else True))
        if tr.sum() and te.sum():
            yield i + 1, d[tr], d[te]


def mcnemar(y, a, b):
    from scipy.stats import chi2 as chi2dist
    a_ok, b_ok = (a == y), (b == y)
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, stat, float(chi2dist.sf(stat, 1))


def main() -> None:
    d = load()
    fe = [c for c in d.columns if c.startswith("fe_")]
    base_rate = d["y"].mean()

    hr("1. SAMPLE")
    print(f"  ticker-days {len(d):,} | tickers {d['symbol'].nunique()} | "
          f"dates {d['date'].nunique()}")
    print(f"  target: next-day return direction (1 = up)")
    print(f"  base rate: {100*base_rate:.1f}% of ticker-days are up days")
    print(f"  a constant 'always up' rule therefore scores "
          f"{100*max(base_rate, 1-base_rate):.1f}% accuracy")

    hr("2. WALK-FORWARD OUT-OF-SAMPLE")
    rows, store = [], {k: {"y": [], "p": [], "s": []} for k in
                       ["Logistic baseline", "Logistic + sentiment",
                        "GBM baseline", "GBM + sentiment"]}

    for k, tr, te in walk_forward(d):
        specs = {
            "Logistic baseline":     (TECH + fe, "logit"),
            "Logistic + sentiment":  (TECH + SENT + fe, "logit"),
            "GBM baseline":          (TECH + fe, "gbm"),
            "GBM + sentiment":       (TECH + SENT + fe, "gbm"),
        }
        line = {"Fold": k, "Train n": len(tr), "Test n": len(te),
                "Test from": te["date"].min().date()}
        for name, (cols, kind) in specs.items():
            Xtr, Xte = tr[cols].astype(float), te[cols].astype(float)
            if kind == "logit":
                sc = StandardScaler().fit(Xtr)          # fitted on train only
                m = LogisticRegression(max_iter=2000, C=1.0)
                m.fit(sc.transform(Xtr), tr["y"])
                s = m.predict_proba(sc.transform(Xte))[:, 1]
            else:
                m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                                   max_depth=3, random_state=42)
                m.fit(Xtr, tr["y"])
                s = m.predict_proba(Xte)[:, 1]
            p = (s > 0.5).astype(int)
            store[name]["y"] += list(te["y"]); store[name]["p"] += list(p)
            store[name]["s"] += list(s)
            line[f"Acc {name}"] = accuracy_score(te["y"], p)
        rows.append(line)
        print(f"  fold {k}: test {len(te):4d} from {te['date'].min().date()}  "
              + "  ".join(f"{n.split()[0][:4]}{'+S' if '+' in n else '  '} "
                          f"{line[f'Acc {n}']:.3f}" for n in specs))

    hr("3. POOLED OUT-OF-SAMPLE PERFORMANCE")
    perf = []
    for name, v in store.items():
        y = np.array(v["y"]); p = np.array(v["p"]); s = np.array(v["s"])
        perf.append({"Model": name, "Accuracy": accuracy_score(y, p),
                     "Balanced accuracy": balanced_accuracy_score(y, p),
                     "Macro F1": f1_score(y, p, average="macro", zero_division=0),
                     "ROC-AUC": roc_auc_score(y, s)})
    y_all = np.array(store["Logistic baseline"]["y"])
    perf.insert(0, {"Model": "Always up (majority rule)",
                    "Accuracy": max(y_all.mean(), 1 - y_all.mean()),
                    "Balanced accuracy": 0.5, "Macro F1": np.nan, "ROC-AUC": np.nan})
    res = pd.DataFrame(perf).set_index("Model").round(4)
    res.to_csv(TAB_DIR / "rq3_performance.csv")
    print(res.to_string())

    hr("4. HYPOTHESIS TEST (McNemar, paired)")
    tests = []
    for base, aug in [("Logistic baseline", "Logistic + sentiment"),
                      ("GBM baseline", "GBM + sentiment")]:
        y = np.array(store[base]["y"])
        a, b = np.array(store[base]["p"]), np.array(store[aug]["p"])
        n01, n10, stat, p = mcnemar(y, a, b)
        better = aug if n01 > n10 else base
        verdict = "reject H0" if p < .05 else "retain H0"
        tests.append({"Comparison": f"{base} vs {aug}",
                      "Only augmented correct": n01, "Only baseline correct": n10,
                      "chi2": round(stat, 3), "p": round(p, 4),
                      "More accurate": better, "Decision at .05": verdict})
        print(f"  {base:20s} vs +sentiment: b={n01:4d} c={n10:4d} "
              f"chi2={stat:6.3f} p={p:.4g} -> {verdict}; more accurate: {better}")
    pd.DataFrame(tests).to_csv(TAB_DIR / "rq3_tests.csv", index=False)

    # ---- figure -----------------------------------------------------------
    f = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    x = np.arange(len(f))
    ax[0].bar(x - 0.2, f["Acc Logistic baseline"], 0.4, color=GREY, label="Technicals only")
    ax[0].bar(x + 0.2, f["Acc Logistic + sentiment"], 0.4, color=BLUE, label="+ sentiment")
    ax[0].axhline(0.5, color=RED, ls=":", lw=1.2, label="coin flip")
    ax[0].set(xlabel="Walk-forward fold", ylabel="Out-of-sample accuracy",
              xticks=x, xticklabels=f["Fold"], ylim=(0.4, 0.62),
              title="(a) Logistic regression by fold")
    ax[0].legend(frameon=False, fontsize=7.5)

    for name, col in [("Logistic baseline", GREY), ("Logistic + sentiment", BLUE),
                      ("GBM + sentiment", RED)]:
        v = store[name]
        fpr, tpr, _ = roc_curve(v["y"], v["s"])
        ax[1].plot(fpr, tpr, lw=1.4, color=col,
                   label=f"{name} (AUC {roc_auc_score(v['y'], v['s']):.3f})")
    ax[1].plot([0, 1], [0, 1], ls=":", color="black", lw=1, label="random (0.500)")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate",
              title="(b) Ranking ability, pooled out-of-sample")
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "fig_rq3_direction.png")
    plt.close(fig)

    hr("5. SUMMARY")
    lb = res.loc["Logistic baseline", "Accuracy"]
    la = res.loc["Logistic + sentiment", "Accuracy"]
    print(f"  logistic: {lb:.4f} -> {la:.4f} ({100*(la-lb):+.2f} pp) with sentiment")
    print(f"  majority rule scores {res.loc['Always up (majority rule)', 'Accuracy']:.4f}")
    print(f"\ntables -> {TAB_DIR.relative_to(C.ROOT)}   "
          f"figures -> {FIG_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
