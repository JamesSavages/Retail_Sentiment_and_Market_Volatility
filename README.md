# Retail Sentiment as a Volatility Early-Warning Signal

Capstone data pipeline (QM640, Walsh College) — StockTwits and news sentiment as
a predictor of next-day equity volatility in the U.S. technology sector.

**Author:** James Savage · **Course:** QM640 Data Analytics Capstone

## What this repo does

Collects three free, public data sources and assembles a ticker-day modelling
panel whose target is **next-day realized volatility (t+1)**:

| Source | Role | Access |
|---|---|---|
| Tiingo | Daily adjusted OHLCV prices, returns, realized volatility (2023-2024) | Free token |
| Alpha Vantage News & Sentiment | Historical per-ticker sentiment feature (RQ2-RQ4, 2023-2024) | Free key, 25 req/day |
| StockTwits | Recent messages with self-tagged Bullish/Bearish labels (RQ1 model validation) | Free, rate-limited, no key |

## Setup (uv)

```bash
# install uv once (https://docs.astral.sh/uv/):
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or: brew install uv  |  winget install astral-sh.uv  |  pipx install uv

uv sync    # creates .venv and installs deps (incl. python-dotenv) from pyproject.toml
```

Create a `.env` file in the project root with your free API keys (the scripts load
it automatically via python-dotenv; StockTwits needs no key):

```
TIINGO_API_KEY=your_tiingo_token            # free at tiingo.com
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key  # free at alphavantage.co
```

`.env` is git-ignored, so your keys are never committed.

## Run order

**1. Collect and build the panel**

```bash
uv run src/test_sources.py             # optional smoke test: one request per source
uv run src/collect_prices.py           # -> data/raw/prices/*.csv        (Tiingo)
uv run src/collect_news_sentiment.py   # -> data/raw/news/*.csv          (resumable; 25/day)
uv run src/collect_stocktwits.py       # -> data/raw/stocktwits/*.csv    (accumulates across runs)
uv run src/build_features.py           # -> data/processed/panel.csv + panel_preview.csv
uv run src/make_data_dictionary.py     # -> data/data_dictionary.csv + .md
```

**2. Verify the panel before modelling**

```bash
uv run src/validate_panel.py           # 27 integrity checks; non-zero exit on any failure
```

Recomputes every derived variable from the raw files and fails if any value
disagrees. Includes a point-in-time test that rebuilds each predictor from
records dated on or before *t*, so look-ahead bias is verified rather than
assumed.

**3. Analysis and modelling**

Add the modelling libraries first:

```bash
uv sync --extra modeling               # transformers, torch, scikit-learn, matplotlib
```

```bash
uv run src/run_eda.py                  # -> docs/figures/*.png, docs/tables/eda_*.csv
uv run src/run_rq1.py                  # FinBERT vs lexicon on the StockTwits corpus
uv run src/run_rq2.py                  # HAR baseline vs sentiment-augmented volatility model
uv run src/run_rq3.py                  # next-day return direction, baseline vs augmented
uv run src/run_rq4.py                  # cost-aware strategy backtest vs buy-and-hold
```

Every figure and table in the report is written to `docs/figures/` and
`docs/tables/` by these scripts, so all reported values are reproducible from the
committed data. `docs/METHOD_WALKTHROUGH.md` explains each stage and every visual
in plain language.

Need a classic `requirements.txt` for a grader? Generate one from the lockfile:
`uv export --format requirements-txt > requirements.txt`

Edit the ticker universe, date window, and volatility estimator in `config.py`.

## Folder structure

```
Retail_Sentiment_and_Market_Volatility/
├── README.md
├── pyproject.toml                # uv project + dependencies
├── config.py                     # universe, dates, rate limits, paths
├── src/
│   ├── test_sources.py           # one-request smoke test for each source
│   ├── collect_prices.py         # Tiingo adjusted OHLCV
│   ├── collect_news_sentiment.py # Alpha Vantage historical sentiment (resumable)
│   ├── collect_stocktwits.py     # recent labelled messages (accumulates + de-dups)
│   ├── build_features.py         # ticker-day panel, volatility, HAR components
│   ├── make_data_dictionary.py   # data dictionary (CSV + MD)
│   ├── validate_panel.py         # 27 integrity checks incl. point-in-time test
│   ├── run_eda.py                # descriptives, figures, VIF, sample adequacy
│   ├── run_rq1.py                # RQ1 FinBERT vs lexicon
│   ├── run_rq2.py                # RQ2 volatility prediction
│   ├── run_rq3.py                # RQ3 return-direction classification
│   └── run_rq4.py                # RQ4 strategy evaluation
├── data/
│   ├── raw/{prices,news,stocktwits}/
│   ├── processed/panel.csv       # full modelling panel
│   ├── processed/panel_preview.csv  # 120-row sample GitHub can render
│   ├── data_dictionary.csv
│   └── data_dictionary.md
├── docs/
│   ├── METHOD_WALKTHROUGH.md     # pipeline and every visual explained
│   ├── figures/                  # all report figures, regenerated by the scripts
│   ├── tables/                   # all report tables as CSV
│   └── screenshots/              # repository evidence for Appendix A
├── notebooks/                    # scratch exploration (analysis lives in src/)
└── report/                       # interim report (.docx and .pdf)
```

## Notes on the design

- **Three-source split.** Tiingo supplies prices (the target and controls);
  Alpha Vantage supplies the historical sentiment feature for RQ2-RQ4; StockTwits
  supplies the recent, human-labelled corpus used only to validate the sentiment
  model in RQ1. StockTwits serves recent messages only, so its columns populate
  for recent dates and are empty across the 2023-2024 modelling window by design.
- **Predictive alignment.** Sentiment features are measured at day *t*; the target
  is realized volatility at *t + 1*, which avoids the same-day sentiment/price
  confound.
- **Volatility target.** Estimated from daily OHLC using Parkinson (default) or
  Garman-Klass, so no intraday data is required; adjusted prices are used so
  corporate actions (e.g., stock splits) do not distort the series.
- **Attention, not just tone.** Daily article count (as `log_articles`) is carried
  alongside the sentiment score, with a `news_missing` flag so that zero-coverage
  days are kept as genuine low-attention observations rather than dropped. The
  Antweiler & Frank (2004) bullishness and agreement indices are implemented in
  `build_features.py` for the StockTwits message panel, which supports RQ1 only
  and is deliberately not merged into the historical RQ2-RQ4 panel.
- **Estimation within ticker.** All panel models carry ticker fixed effects. Pooled
  comparisons reverse the sign of the attention effect (a Simpson's paradox
  documented in the report), so only within-ticker variation is informative.

## Data source history (for the Limitations section)

An initial price source (Stooq) was found to block automated access, and Alpha
Vantage's full-history price endpoint proved premium-gated; Tiingo was adopted
after validating access. This deliberate evaluation of sources supports
reproducibility.