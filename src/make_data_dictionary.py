"""
Generate the data dictionary (mandatory synopsis artifact) for the processed
panel and the raw sources, in both CSV and Markdown.

Run:
    python src/make_data_dictionary.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# variable | source | type | unit | role | description
SCHEMA = [
    ("symbol", "All", "string", "-", "key", "Stock ticker symbol (e.g., AAPL)."),
    ("date", "All", "date", "YYYY-MM-DD", "key", "Trading date (US/Eastern)."),
    ("close", "Tiingo", "float", "USD", "raw", "Daily closing price."),
    ("log_return", "Derived", "float", "log points", "control", "ln(close_t / close_{t-1})."),
    ("abs_return", "Derived", "float", "log points", "control", "Absolute daily log return."),
    ("log_volume", "Derived", "float", "ln(shares)", "control", "log(1 + daily share volume)."),
    ("rv", "Derived", "float", "daily sigma", "control", "Realized volatility at t (range-based estimator)."),
    ("rv_lag1", "Derived", "float", "daily sigma", "control", "Realized volatility at t-1."),
    ("rv_next", "Derived", "float", "daily sigma", "TARGET", "Next-day realized volatility (t+1) - primary Y."),
    ("news_sent_wmean", "Alpha Vantage", "float", "-1 to +1", "predictor", "Relevance-weighted mean ticker news sentiment."),
    ("news_article_count", "Alpha Vantage", "int", "articles", "predictor", "Number of news articles on the ticker-day."),
]

# Note: the StockTwits labelled corpus (data/raw/stocktwits/*.csv) is the RQ1
# sentiment-model validation set, analysed at the message level (fields: id,
# created_at, symbol, body, sentiment_basic). It is recent-only and is not part
# of the historical RQ2-RQ4 panel documented above.


def main() -> None:
    df = pd.DataFrame(SCHEMA, columns=["variable", "source", "type", "unit", "role", "description"])
    df.to_csv(C.DICT_PATH, index=False)
    print(f"[dict] wrote {C.DICT_PATH} ({len(df)} variables)")

    md = C.DATA / "data_dictionary.md"
    with open(md, "w") as fh:
        fh.write("# Data Dictionary\n\n")
        fh.write("| Variable | Source | Type | Unit | Role | Description |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            fh.write(f"| `{r.variable}` | {r.source} | {r.type} | {r.unit} | {r.role} | {r.description} |\n")
    print(f"[dict] wrote {md}")


if __name__ == "__main__":
    main()