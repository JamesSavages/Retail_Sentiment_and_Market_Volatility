# Data Dictionary

| Variable | Source | Type | Unit | Role | Description |
|---|---|---|---|---|---|
| `symbol` | All | string | - | key | Stock ticker symbol (e.g., AAPL). |
| `date` | All | date | YYYY-MM-DD | key | Trading date (US/Eastern). |
| `close` | Tiingo | float | USD | raw | Daily closing price. |
| `log_return` | Derived | float | log points | control | ln(close_t / close_{t-1}). |
| `abs_return` | Derived | float | log points | control | Absolute daily log return. |
| `log_volume` | Derived | float | ln(shares) | control | log(1 + daily share volume). |
| `rv` | Derived | float | daily sigma | control | Realized volatility at t (range-based estimator). |
| `rv_lag1` | Derived | float | daily sigma | control | Realized volatility at t-1. |
| `rv_next` | Derived | float | daily sigma | TARGET | Next-day realized volatility (t+1) - primary Y. |
| `news_sent_wmean` | Alpha Vantage | float | -1 to +1 | predictor | Relevance-weighted mean ticker news sentiment. |
| `news_article_count` | Alpha Vantage | int | articles | predictor | Number of news articles on the ticker-day. |
