# Retail Sentiment as a Volatility Early-Warning Signal

Capstone data pipeline (QM640, Walsh College) — StockTwits and news sentiment as
a predictor of next-day equity volatility in the U.S. technology sector.

**Author:** James Savage · **Course:** QM640 Data Analytics Capstone

## What this repo does

Collects three public data sources and assembles a ticker-day modelling panel
whose target is **next-day realized volatility (t+1)**:

| Source | Role | Access |
|---|---|---|
| Stooq | Daily OHLCV prices, returns, realized volatility (2023-2024) | Free, no key |
| StockTwits | Recent messages with self-tagged Bullish/Bearish labels (RQ1 model validation) | Free, rate-limited |
| Alpha Vantage News & Sentiment | Historical per-ticker sentiment feature (RQ2-RQ4, 2023-2024) | Free key, 25 req/day |

## Setup (uv)

```bash
# install uv once (https://docs.astral.sh/uv/):
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or: brew install uv  |  winget install astral-sh.uv  |  pipx install uv

uv sync                                # creates .venv and installs deps from pyproject.toml
export ALPHAVANTAGE_API_KEY=your_free_key   # from alphavantage.co (or put it in a .env)
```

## Run order

```bash
uv run src/collect_prices.py           # -> data/raw/prices/*.csv
uv run src/collect_stocktwits.py       # -> data/raw/stocktwits/*.csv
uv run src/collect_news_sentiment.py   # -> data/raw/news/*.csv  (resumable; 25/day)
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
├── pyproject.toml             # uv project + dependencies
├── config.py                     # universe, dates, rate limits, paths
├── src/
│   ├── collect_prices.py         # Stooq OHLCV
│   ├── collect_stocktwits.py     # recent labelled messages
│   ├── collect_news_sentiment.py # Alpha Vantage historical sentiment
│   ├── build_features.py         # ticker-day panel + volatility + AF indices
│   └── make_data_dictionary.py   # data dictionary (CSV + MD)
├── data/
│   ├── raw/{prices,stocktwits,news}/
│   ├── processed/panel.csv
│   ├── data_dictionary.csv
│   └── data_dictionary.md
├── notebooks/                    # EDA / modelling (later stages)
└── docs/screenshots/             # screenshots referenced in the synopsis
```

## Notes on the design

- **Predictive alignment.** Sentiment features are measured at day *t*; the target
  is realized volatility at *t + 1*. Post-close StockTwits messages are rolled to
  the next trading day. This avoids the same-day sentiment/price confound.
- **Volatility target.** Estimated from daily OHLC using Parkinson (default) or
  Garman-Klass, so no intraday data is required.
- **Sentiment dispersion.** Captured via the Antweiler & Frank (2004) bullishness
  and agreement indices, which the literature links to volatility and volume.
