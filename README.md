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

```bash
uv run src/test_sources.py             # optional smoke test: one request per source
uv run src/collect_prices.py           # -> data/raw/prices/*.csv        (Tiingo)
uv run src/collect_news_sentiment.py   # -> data/raw/news/*.csv          (resumable; 25/day)
uv run src/collect_stocktwits.py       # -> data/raw/stocktwits/*.csv    (accumulates across runs)
uv run src/build_features.py           # -> data/processed/panel.csv
uv run src/make_data_dictionary.py     # -> data/data_dictionary.csv + .md
```

When you reach the RQ1 sentiment-model stage, add the modelling libraries:

```bash
uv sync --extra modeling               # transformers, torch, scikit-learn, matplotlib
```

Need a classic `requirements.txt` for a grader? Generate one from the lockfile:
`uv export --format requirements-txt > requirements.txt`

Edit the ticker universe, date window, and volatility estimator in `config.py`.

## Folder structure

```
stocktwits-volatility-capstone/
├── README.md
├── pyproject.toml                # uv project + dependencies
├── config.py                     # universe, dates, rate limits, paths
├── src/
│   ├── test_sources.py           # one-request smoke test for each source
│   ├── collect_prices.py         # Tiingo adjusted OHLCV
│   ├── collect_news_sentiment.py # Alpha Vantage historical sentiment (resumable)
│   ├── collect_stocktwits.py     # recent labelled messages (accumulates + de-dups)
│   ├── build_features.py         # ticker-day panel + volatility + AF indices
│   └── make_data_dictionary.py   # data dictionary (CSV + MD)
├── data/
│   ├── raw/{prices,news,stocktwits}/
│   ├── processed/panel.csv
│   ├── data_dictionary.csv
│   └── data_dictionary.md
├── notebooks/                    # EDA / modelling (later stages)
└── docs/screenshots/             # screenshots referenced in the synopsis
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
- **Sentiment dispersion.** Captured via the Antweiler & Frank (2004) bullishness
  and agreement indices, which the literature links to volatility and volume.

## Data source history (for the Limitations section)

An initial price source (Stooq) was found to block automated access, and Alpha
Vantage's full-history price endpoint proved premium-gated; Tiingo was adopted
after validating access. This deliberate evaluation of sources supports
reproducibility.