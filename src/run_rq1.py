"""
RQ1: How accurately can FinBERT classify StockTwits message sentiment, compared
with a lexicon baseline and users' own bullish/bearish tags?

    H0: accuracy(FinBERT) <= accuracy(lexicon)
    H1: accuracy(FinBERT) >  accuracy(lexicon)
    Test: McNemar's test (paired -- both models score identical messages)

WHY ACCURACY ALONE IS NOT REPORTED
    82% of the labelled corpus is tagged Bullish. A classifier that answers
    "bullish" to everything therefore scores ~82% accuracy while carrying no
    information at all. That trivial rule is computed and reported as the
    majority-class floor, and the headline metrics are BALANCED ACCURACY and
    MACRO-F1, which weight the two classes equally.

CLEANING
    The raw stream contains promotional spam and messages that are tagged to one
    ticker while discussing another. Both would corrupt the label, so both are
    removed before scoring. Counts are reported.

Outputs:
    docs/tables/rq1_performance.csv       headline comparison (-> report table)
    docs/tables/rq1_confusion.csv         confusion matrix per model
    docs/tables/rq1_cleaning.csv          corpus cleaning log
    docs/figures/fig_rq1_confusion.png    confusion matrices
    docs/figures/fig_rq1_roc.png          ROC curves

Run:
    uv run src/run_rq1.py                 full run
    uv run src/run_rq1.py --no-finbert    lexicon baselines only (fast)
    uv run src/run_rq1.py --limit 500     smoke test on a subsample
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, confusion_matrix)  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

FIG_DIR = C.ROOT / "docs" / "figures"
TAB_DIR = C.ROOT / "docs" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FINBERT_MODEL = "ProsusAI/finbert"
MAX_CASHTAGS = 3          # more than this suggests a promotional blast
BLUE, GREY, RED = "#2c6fbb", "#4d4d4d", "#c0392b"

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ----------------------------------------------------------------------
# Loughran-McDonald finance word list
# ----------------------------------------------------------------------
# The full Loughran-McDonald master dictionary is not redistributed with this
# repository. If you place it at data/external/LoughranMcDonald_MasterDictionary.csv
# the script uses it in full. Otherwise it falls back to the compact word list
# below, drawn from the same Positive and Negative categories, and says so in the
# output so the report can describe exactly what was used.
LM_FALLBACK_POS = """
able achieve achieved achievement advantage advantages beneficial benefit benefits
best better boost boosted breakthrough compliment strength strengthen strengthened
strong stronger strongest succeed success successful successfully superior upturn
gain gains gained profitable profitability outperform outperformed exceeded exceeds
opportunity opportunities improve improved improvement improvements favorable
""".split()

LM_FALLBACK_NEG = """
adverse adversely against bad bankruptcy breach concern concerns crisis critical
damage damages decline declined declines deficit deteriorate deteriorated difficult
difficulty disappointing downgrade downturn fail failed failure failures fraud
impair impaired impairment inadequate ineffective investigation lawsuit litigation
loss losses lost negative penalty poor recall restructuring risk risks
shortfall slowdown suspend suspended terminate terminated threat unable
underperform violation volatile weak weakened weakness weaknesses worse worst
""".split()


def load_lm() -> tuple[set, set, str]:
    p = C.DATA / "external" / "LoughranMcDonald_MasterDictionary.csv"
    if p.exists():
        d = pd.read_csv(p)
        cols = {c.lower(): c for c in d.columns}
        w, pos, neg = cols.get("word"), cols.get("positive"), cols.get("negative")
        if w and pos and neg:
            P = set(d.loc[d[pos] > 0, w].str.lower())
            N = set(d.loc[d[neg] > 0, w].str.lower())
            return P, N, f"full master dictionary ({len(P)} pos / {len(N)} neg)"
    return (set(LM_FALLBACK_POS), set(LM_FALLBACK_NEG),
            f"compact fallback list ({len(LM_FALLBACK_POS)} pos / "
            f"{len(LM_FALLBACK_NEG)} neg) -- place the master dictionary at "
            f"data/external/ to use it in full")


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------
CASHTAG = re.compile(r"\$[A-Za-z][A-Za-z.\-]{0,6}")
URL = re.compile(r"https?://\S+|www\.\S+")
PROMO = re.compile(r"\b(?:yearn|airdrop|presale|telegram|whatsapp|join now|dm me|"
                   r"free signals|100x|giveaway)\b", re.I)


def load_corpus(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for f in glob.glob(str(C.RAW_ST / "*.csv")):
        if os.path.getsize(f) == 0:
            continue
        d = pd.read_csv(f)
        if not d.empty:
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    log = [("Messages collected", len(raw), "")]

    raw = raw.drop_duplicates(subset="id")
    log.append(("After de-duplication by message id", len(raw), "repeat pulls overlap"))

    df = raw[raw["sentiment_basic"].isin(["Bullish", "Bearish"])].copy()
    log.append(("With a self-tagged Bullish/Bearish label", len(df),
                f"{100*len(df)/len(raw):.1f}% of the corpus is labelled"))

    df["body"] = df["body"].astype(str)

    # Promotional spam: the tag does not describe the ticker's prospects.
    spam = df["body"].str.contains(PROMO) | (
        df["body"].str.contains(URL) & (df["body"].str.len() < 90))
    df = df[~spam]
    log.append(("Less promotional spam", len(df), f"{int(spam.sum())} removed"))

    # Multi-ticker blasts: a message listing many tickers is not expressing a
    # view about any one of them. A message that never mentions its own ticker
    # cannot be assumed to be about it either. Both are dropped; a message that
    # names its ticker alongside one or two others is kept, since discussing a
    # peer is normal and the self-tag still refers to the tagged symbol.
    n_tags = df["body"].str.count(CASHTAG)
    names_self = [bool(re.search(rf"\${re.escape(s)}\b", b, re.I))
                  for s, b in zip(df["symbol"], df["body"])]
    off_topic = (n_tags > MAX_CASHTAGS) | (~pd.Series(names_self, index=df.index))
    df = df[~off_topic]
    log.append(("Less off-topic multi-ticker messages", len(df),
                f"{int(off_topic.sum())} removed (more than "
                f"{MAX_CASHTAGS} cashtags, or own ticker not mentioned)"))

    # Model input: strip cashtags and URLs so the classifier reads the prose.
    df["text"] = (df["body"].str.replace(URL, " ", regex=True)
                            .str.replace(CASHTAG, " ", regex=True)
                            .str.replace(r"&amp;", "&", regex=True)
                            .str.replace(r"&#39;", "'", regex=True)
                            .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[df["text"].str.len() >= 10]
    log.append(("Less messages under 10 characters of text", len(df),
                "too short to classify"))

    df["y"] = (df["sentiment_basic"] == "Bullish").astype(int)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42)
        log.append(("Subsampled for a smoke test", len(df), "--limit was set"))

    return df.reset_index(drop=True), pd.DataFrame(
        log, columns=["Step", "Messages Remaining", "Note"])


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def score_vader(texts) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    return np.array([an.polarity_scores(t)["compound"] for t in texts])


def score_lm(texts, pos: set, neg: set) -> np.ndarray:
    out = []
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        if not w:
            out.append(0.0)
            continue
        p = sum(x in pos for x in w)
        n = sum(x in neg for x in w)
        out.append(0.0 if p + n == 0 else (p - n) / (p + n))
    return np.array(out)


def score_finbert(texts, batch: int = 32) -> np.ndarray:
    """Return P(positive) - P(negative) for each message."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    mdl.eval()

    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    ip, ineg = lab.get("positive", 0), lab.get("negative", 1)

    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        with torch.no_grad():
            pr = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
        out.append(pr[:, ip] - pr[:, ineg])
        if (i // batch) % 20 == 0:
            print(f"    FinBERT {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """
    Threshold that maximises balanced accuracy, chosen on the TRAINING half only.

    Needed because both scorers emit a continuous sentiment value and the naive
    cut at zero assumes a balanced corpus. This one is 82% bullish, so a
    zero cut labels far too many messages bearish. Calibrating on held-out-from-
    test data is the honest way to give the lexicon its best shot; evaluating on
    the same data used to pick the threshold would flatter it.
    """
    cand = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
    scores = [balanced_accuracy_score(y, (score > t).astype(int)) for t in cand]
    return float(cand[int(np.argmax(scores))])


def evaluate(name: str, y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average=None,
                                                 labels=[0, 1], zero_division=0)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Macro F1": f1_score(y, pred, average="macro", zero_division=0),
        "F1 (bearish)": f[0], "F1 (bullish)": f[1],
        "Recall (bearish)": r[0], "Recall (bullish)": r[1],
        "ROC-AUC": roc_auc_score(y, score) if score is not None else np.nan,
    }


def paired_bootstrap(y, s_a, s_b, p_a, p_b, n_boot: int = 5000, seed: int = 42):
    """
    Paired bootstrap on the metric difference (B minus A), resampling messages.

    McNemar tests ACCURACY, which is the wrong criterion on an 82/18 corpus: a
    model can be significantly less accurate while being markedly better at the
    minority class. Step 4 of the research design permits a paired bootstrap as
    the alternative, so balanced accuracy and ROC-AUC are tested that way. Both
    models score the identical resampled messages, so the comparison stays paired.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {}
    for name, fn, a, b in [
        ("Balanced accuracy", balanced_accuracy_score, p_a, p_b),
        ("ROC-AUC", roc_auc_score, s_a, s_b),
    ]:
        if a is None or b is None:
            continue
        obs = fn(y, b) - fn(y, a)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:        # need both classes present
                diffs[i] = np.nan
                continue
            diffs[i] = fn(y[idx], b[idx]) - fn(y[idx], a[idx])
        diffs = diffs[~np.isnan(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # Two-sided p: proportion of resamples on the far side of zero. A
        # bootstrap p can never truly be zero -- it is bounded below by the
        # resolution of the resampling, so it is floored at 1/n_boot and
        # reported as "< that value" rather than as 0.
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        floored = p < 1 / len(diffs)
        out[name] = {"observed_diff": obs, "ci_low": lo, "ci_high": hi,
                     "p": max(p, 1 / len(diffs)), "p_floored": floored}
    return out


def mcnemar(y, a, b) -> tuple[int, int, float, float]:
    """Compare two classifiers on the same items. Returns b, c, chi2, p."""
    from scipy.stats import chi2 as chi2dist
    a_ok, b_ok = (a == y), (b == y)
    n01 = int((~a_ok & b_ok).sum())    # only B correct
    n10 = int((a_ok & ~b_ok).sum())    # only A correct
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, stat, float(chi2dist.sf(stat, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-finbert", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df, clean_log = load_corpus(args.limit)
    clean_log.to_csv(TAB_DIR / "rq1_cleaning.csv", index=False)

    hr("1. CORPUS")
    print(clean_log.to_string(index=False))
    y = df["y"].values
    base_rate = y.mean()
    print(f"\n  labelled messages used : {len(df):,}")
    print(f"  bullish                : {y.sum():,} ({100*base_rate:.1f}%)")
    print(f"  bearish                : {(1-y).sum():,} ({100*(1-base_rate):.1f}%)")
    print(f"  median length          : {df['text'].str.len().median():.0f} characters")

    # --- train / test split ------------------------------------------------
    # Thresholds are chosen on train; every reported metric is computed on test.
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                      random_state=42, stratify=y)
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"\n  train {len(idx_tr):,} / test {len(idx_te):,} "
          f"(stratified, seed 42; all metrics below are TEST-set)")

    results, preds, scores = [], {}, {}

    def add(name: str, s_all: np.ndarray | None, calibrate: bool = True):
        """Score a model on the test set, at the natural and calibrated cut."""
        if s_all is None:                                   # majority-class rule
            pr = np.full_like(y_te, int(y_tr.mean() >= 0.5))
            results.append(evaluate(name, y_te, None, pr))
            preds[name] = pr
            return
        s_te = s_all[idx_te]
        pr0 = (s_te > 0).astype(int)
        results.append(evaluate(f"{name} (cut at 0)", y_te, s_te, pr0))
        preds[f"{name} (cut at 0)"] = pr0
        scores[name] = s_te
        if calibrate:
            t = best_threshold(y_tr, s_all[idx_tr])
            prc = (s_te > t).astype(int)
            results.append(evaluate(f"{name} (calibrated cut {t:+.3f})",
                                    y_te, s_te, prc))
            preds[name] = prc                                # calibrated is primary
        r = results[-1]
        print(f"  {name:28s} acc {r['Accuracy']:.3f}  "
              f"bal-acc {r['Balanced accuracy']:.3f}  "
              f"macro-F1 {r['Macro F1']:.3f}  AUC {r['ROC-AUC']:.3f}")

    add("Majority class (always bullish)", None)

    # --- lexicon baselines -------------------------------------------------
    hr("2. LEXICON BASELINES")
    pos, neg, lm_src = load_lm()
    print(f"  Loughran-McDonald: {lm_src}\n")
    v = score_vader(df["text"])
    l = score_lm(df["text"], pos, neg)
    combo = np.where(l != 0, 0.5 * v + 0.5 * l, v)   # LM contributes only when it fires
    add("VADER only", v)
    add("VADER + Loughran-McDonald", combo)

    # --- FinBERT -----------------------------------------------------------
    if not args.no_finbert:
        hr("3. FINBERT")
        try:
            fb = score_finbert(df["text"])
            add("FinBERT", fb)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  FinBERT unavailable ({type(exc).__name__}: {exc})")
            print("  Install with: uv sync --extra modeling")

    y = y_te                          # everything below is evaluated on test
    res = pd.DataFrame(results).set_index("Model").round(4)
    res.to_csv(TAB_DIR / "rq1_performance.csv")

    hr("4. RESULTS")
    print(res.to_string())
    print(f"\n  NOTE: the majority-class floor is {100*max(base_rate, 1-base_rate):.1f}% "
          f"accuracy. Any model at or below that carries no information,\n"
          f"  which is why balanced accuracy and macro-F1 are the headline metrics.")

    # --- hypothesis test ---------------------------------------------------
    if "FinBERT" in preds:
        hr("5. HYPOTHESIS TEST (McNemar, paired)")
        rows = []
        for other in ["VADER only", "VADER + Loughran-McDonald",
                      "Majority class (always bullish)"]:
            if other not in preds:
                continue
            n01, n10, stat, p = mcnemar(y, preds[other], preds["FinBERT"])
            better = "FinBERT" if n01 > n10 else other
            verdict = ("reject H0 (accuracy differs)" if p < .05
                       else "retain H0 (no accuracy difference)")
            rows.append({"Comparison": f"FinBERT vs {other}",
                         "Only FinBERT correct": n01, "Only other correct": n10,
                         "chi2": round(stat, 3), "p": p,
                         "More accurate": better, "Decision at .05": verdict})
            print(f"  FinBERT vs {other:32s} b={n01:4d} c={n10:4d} "
                  f"chi2={stat:7.2f} p={p:.3g}")
            print(f"      -> {verdict}; MORE ACCURATE: {better}")
        pd.DataFrame(rows).to_csv(TAB_DIR / "rq1_mcnemar.csv", index=False)

        # Accuracy is the wrong criterion at an 82/18 base rate, so the
        # balanced-accuracy and ranking comparisons are tested by paired bootstrap.
        hr("6. PAIRED BOOTSTRAP (balanced accuracy and ranking)")
        brows = []
        for other in ["VADER only", "VADER + Loughran-McDonald"]:
            if other not in preds:
                continue
            res_b = paired_bootstrap(y, scores.get(other), scores.get("FinBERT"),
                                     preds[other], preds["FinBERT"])
            for metric, r in res_b.items():
                sig = "significant" if r["p"] < .05 else "not significant"
                fav = "FinBERT" if r["observed_diff"] > 0 else other
                ptxt = ("< %.4f" % r["p"]) if r["p_floored"] else ("%.4f" % r["p"])
                brows.append({"Comparison": f"FinBERT vs {other}", "Metric": metric,
                              "Difference (FinBERT - other)": round(r["observed_diff"], 4),
                              "95% CI low": round(r["ci_low"], 4),
                              "95% CI high": round(r["ci_high"], 4),
                              "p": ptxt, "Favours": fav, "Decision at .05": sig})
                print(f"  {other:28s} {metric:18s} "
                      f"diff {r['observed_diff']:+.4f} "
                      f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                      f"p={ptxt} -> favours {fav}, {sig}")
        pd.DataFrame(brows).to_csv(TAB_DIR / "rq1_bootstrap.csv", index=False)

    # --- figures -----------------------------------------------------------
    show = [m for m in ["VADER + Loughran-McDonald", "FinBERT"] if m in preds]
    if show:
        fig, ax = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.2),
                               constrained_layout=True)
        ax = np.atleast_1d(ax)
        for a, m in zip(ax, show):
            cm = confusion_matrix(y, preds[m], labels=[0, 1])
            a.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                           fontsize=11,
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
            a.set(xticks=[0, 1], yticks=[0, 1],
                  xticklabels=["pred. bearish", "pred. bullish"],
                  yticklabels=["actual bearish", "actual bullish"],
                  title=f"{m}\nbalanced accuracy "
                        f"{balanced_accuracy_score(y, preds[m]):.3f}")
        fig.savefig(FIG_DIR / "fig_rq1_confusion.png")
        plt.close(fig)

        cmrows = []
        for m in preds:
            cm = confusion_matrix(y, preds[m], labels=[0, 1])
            cmrows.append({"Model": m, "TN (bearish correct)": cm[0, 0],
                           "FP": cm[0, 1], "FN": cm[1, 0],
                           "TP (bullish correct)": cm[1, 1]})
        pd.DataFrame(cmrows).to_csv(TAB_DIR / "rq1_confusion.csv", index=False)

    if scores:
        fig, a = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
        for m, s in scores.items():
            fpr, tpr, _ = roc_curve(y, s)
            a.plot(fpr, tpr, lw=1.6,
                   label=f"{m} (AUC {roc_auc_score(y, s):.3f})")
        a.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1, label="random (AUC 0.500)")
        a.set(xlabel="False positive rate", ylabel="True positive rate",
              title="RQ1: ranking ability (ROC)")
        a.legend(frameon=False, fontsize=7.5, loc="lower right")
        a.spines[["top", "right"]].set_visible(False)
        fig.savefig(FIG_DIR / "fig_rq1_roc.png")
        plt.close(fig)

    print(f"\ntables -> {TAB_DIR.relative_to(C.ROOT)}   "
          f"figures -> {FIG_DIR.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
