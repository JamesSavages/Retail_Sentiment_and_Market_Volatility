"""
Generate the data dictionary (mandatory report artifact) for the processed panel,
in CSV and Markdown.

The column set matches the interim/final report template exactly:

    Variable Name | Definition | Datatype | Allowed Values/Range |
    Missing Values Handling | Notes

Two extra columns (Source, Role) are carried as well, because they are useful in
the report and cost nothing. Ranges are read from the live panel where possible,
so the dictionary cannot drift out of step with the data.

Run:
    uv run src/make_data_dictionary.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# name | source | dtype | role | definition | allowed/range | missing handling | notes
SCHEMA = [
    ("symbol", "All", "string", "key",
     "Stock ticker symbol identifying the equity.",
     "24 tickers; see config.TICKERS",
     "None permitted; row is invalid without it.",
     "Analysis sample restricted to the 8 tickers in config.NEWS_TICKERS."),

    ("date", "All", "date", "key",
     "Trading date (US/Eastern) to which the observation applies.",
     "2023-01-03 to 2024-12-31, NYSE trading days",
     "None permitted; row is invalid without it.",
     "Non-trading days are absent by construction, not imputed."),

    ("close", "Tiingo", "float", "raw",
     "Split- and dividend-adjusted daily closing price.",
     "> 0 USD",
     "None observed; a null would indicate a failed collection.",
     "Adjusted series (adjClose), so corporate actions do not create false volatility."),

    ("log_return", "Derived", "float", "control",
     "Daily logarithmic return, ln(close_t / close_{t-1}).",
     "approx -0.5 to +0.5 log points",
     "Null on each ticker's first trading day; row excluded from modelling.",
     "Used as a lagged control, never contemporaneously with the target."),

    ("abs_return", "Derived", "float", "control",
     "Absolute daily log return, |log_return|, a return-magnitude control.",
     ">= 0 log points",
     "Null wherever log_return is null; row excluded.",
     "Correlated with rv; retained but monitored via VIF."),

    ("log_volume", "Derived", "float", "control",
     "Log of adjusted daily share volume, log(1 + volume).",
     "> 0 ln(shares)",
     "None observed.",
     "log1p used so a zero-volume day maps to 0 rather than -inf."),

    ("rv", "Derived", "float", "control",
     "Realized volatility at t from the daily high-low range (Parkinson estimator).",
     "> 0 daily sigma",
     "None observed.",
     "Known at the close of day t, so admissible as a predictor of t+1. "
     "Serves as the daily component of the HAR baseline."),

    ("rv_lag1", "Derived", "float", "control",
     "Realized volatility at t-1.",
     "> 0 daily sigma",
     "Null on each ticker's first trading day; row excluded.",
     "Retained for continuity with the synopsis specification."),

    ("rv_w", "Derived", "float", "control",
     "Weekly HAR component: mean realized volatility over t-4 to t inclusive.",
     "> 0 daily sigma",
     "Null for the first 4 trading days per ticker; rows excluded.",
     "Window ends at t because rv_t is observable at that day's close."),

    ("rv_m", "Derived", "float", "control",
     "Monthly HAR component: mean realized volatility over t-21 to t inclusive.",
     "> 0 daily sigma",
     "Null for the first 21 trading days per ticker; rows excluded.",
     "Binding constraint on sample size: costs 21 ticker-days per ticker."),

    ("rv_next", "Derived", "float", "TARGET",
     "Next-day realized volatility (t+1). Primary dependent variable Y.",
     "> 0 daily sigma",
     "Null on each ticker's final trading day; row excluded.",
     "The only column that uses information dated after t, by design."),

    ("news_sent_wmean", "Alpha Vantage", "float", "predictor",
     "Relevance-weighted mean news sentiment for the ticker on day t.",
     "-1 (bearish) to +1 (bullish)",
     "Set to 0.0 (neutral) on queried tickers with no articles, flagged by "
     "news_missing. Left null for tickers never queried.",
     "Weights are Alpha Vantage relevance scores; unweighted mean if all weights are 0."),

    ("news_article_count", "Alpha Vantage", "int", "predictor",
     "Number of news articles referencing the ticker on day t; attention proxy.",
     ">= 0 articles",
     "Set to 0 on queried tickers with no articles (a genuine low-attention "
     "observation). Left null for tickers never queried.",
     "Zero-article days are retained deliberately; dropping them truncates the "
     "attention measure at 1 and removes the low-attention contrast."),

    ("news_missing", "Derived", "int", "flag",
     "1 if the ticker-day had no news articles, 0 if it had at least one.",
     "{0, 1}",
     "Null for tickers never queried for news.",
     "Lets the model separate genuinely neutral coverage from absent coverage."),

    ("log_articles", "Derived", "float", "predictor",
     "Log attention proxy, log(1 + news_article_count).",
     ">= 0",
     "Null wherever news_article_count is null.",
     "Well defined only because zero-article days are retained."),
]

COLUMNS = ["variable", "source", "datatype", "role", "definition",
           "allowed_values_range", "missing_values_handling", "notes"]

# Report-facing headers, matching the template's data-dictionary table.
HEADERS = ["Variable Name", "Source", "Datatype", "Role", "Definition",
           "Allowed Values/Range", "Missing Values Handling", "Notes"]


def observed_ranges() -> dict[str, str]:
    """Read actual min/max from the built panel so the dictionary cannot drift."""
    path = C.PROCESSED / "panel.csv"
    if not path.exists():
        return {}
    p = pd.read_csv(path)
    out = {}
    for col in p.columns:
        s = pd.to_numeric(p[col], errors="coerce").dropna()
        if len(s):
            out[col] = f"observed {s.min():.4g} to {s.max():.4g}"
    return out


def main() -> None:
    df = pd.DataFrame(SCHEMA, columns=COLUMNS)

    obs = observed_ranges()
    if obs:
        df["observed_range"] = df["variable"].map(obs).fillna("-")
        print(f"[dict] observed ranges read from panel.csv for {len(obs)} columns")

    df.to_csv(C.DICT_PATH, index=False)
    print(f"[dict] wrote {C.DICT_PATH} ({len(df)} variables)")

    md = C.DATA / "data_dictionary.md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("# Data Dictionary\n\n")
        fh.write("Panel: `data/processed/panel.csv` (one row per ticker-day).\n\n")
        fh.write("| " + " | ".join(HEADERS) + " |\n")
        fh.write("|" + "---|" * len(HEADERS) + "\n")
        for _, r in df.iterrows():
            cells = [f"`{r.variable}`", r.source, r.datatype, r.role, r.definition,
                     r.allowed_values_range, r.missing_values_handling, r.notes]
            fh.write("| " + " | ".join(str(c) for c in cells) + " |\n")
        fh.write(
            "\n**Note.** The StockTwits labelled corpus "
            "(`data/raw/stocktwits/*.csv`: id, created_at, symbol, body, "
            "sentiment_basic) is the RQ1 sentiment-model validation set, analysed "
            "at message level. It is recent-only and is deliberately not merged "
            "into the historical RQ2-RQ4 panel documented above. No "
            "user-identifying fields are retained.\n"
        )
    print(f"[dict] wrote {md}")


if __name__ == "__main__":
    main()
