"""
Central configuration for the StockTwits Volatility Early-Warning capstone.

Everything that might change (ticker universe, dates, rate limits, output paths)
lives here so the collection and feature scripts stay clean.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_ST = DATA / "raw" / "stocktwits"
RAW_PX = DATA / "raw" / "prices"
RAW_NEWS = DATA / "raw" / "news"
PROCESSED = DATA / "processed"
DICT_PATH = DATA / "data_dictionary.csv"
for _p in (RAW_ST, RAW_PX, RAW_NEWS, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Study universe and window
# ----------------------------------------------------------------------
# ~24 high-liquidity U.S. technology equities (high StockTwits chatter).
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "AVGO", "ADBE", "CRM", "ORCL", "INTC", "QCOM", "CSCO", "NFLX",
    "PLTR", "MU", "TXN", "IBM", "NOW", "UBER", "SHOP", "ARM",
]

# For the free Alpha Vantage tier (25 requests/day), the *historical* news
# panel is collected on a reduced set first; expand as your quota allows.
NEWS_TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "META", "PLTR"]

START_DATE = "2023-01-01"   # study window start (prices / news)
END_DATE = "2024-12-31"     # study window end

# ----------------------------------------------------------------------
# Collection behaviour
# ----------------------------------------------------------------------
# StockTwits public stream returns only *recent* messages. Used to build the
# labelled corpus for the RQ1 sentiment-model validation and for screenshots.
STOCKTWITS_PAGES_PER_TICKER = 5      # ~30 messages/page; keep small for a sample
STOCKTWITS_DELAY_SEC = 2.0           # be polite; respect rate limits
STOCKTWITS_USER_AGENT = "capstone-research/1.0 (academic use)"

# Alpha Vantage: set your free key in the environment as ALPHAVANTAGE_API_KEY.
AV_DELAY_SEC = 15.0                  # free tier ~5/min; 15s is safe
AV_VOLATILITY_ESTIMATOR = "parkinson"   # "parkinson" | "garman_klass" | "rolling_std"
ROLLING_STD_WINDOW = 5               # trading days, if estimator == rolling_std

# Assign StockTwits messages posted after the US market close (16:00 ET) to the
# next trading day, so post-close chatter informs t+1 (predictive alignment).
ROLL_POST_CLOSE_TO_NEXT_DAY = True
