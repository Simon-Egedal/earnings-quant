# Earnings Quant

Earnings Quant is a Python research and event-backtesting project for estimating a company's next quarterly financial results, comparing those estimates with analyst consensus, and predicting the stock's three-session abnormal return around earnings. It produces `LONG`, `SHORT`, or `NO_TRADE` research signals. It does **not** connect to a broker or execute trades.

The implementation favors time-correct data and simple baselines over model complexity. It supports hundreds of US companies, caches raw responses, writes tabular data as Parquet, records artifact metadata in SQLite, and runs on Windows with PowerShell.

## Architecture

```text
SEC Company Facts + Yahoo prices/earnings
                  |
                  v
        cached normalized Parquet
                  |
                  v
 point-in-time event rows (filed_at < event time)
          |                       |
          v                       v
 Model A: EPS/revenue/       historical event returns
 margin/FCF forecasts              |
          |                        |
          +---- forecast surprises-+
                         |
                         v
          Model B: abnormal 3D return + P(up)
                         |
                         v
      walk-forward evaluation -> event backtest -> scanner
```

Key modules:

- `src/providers`: rate-limited SEC Company Facts and defensive yfinance adapters.
- `src/data`: S&P 500 universe discovery, collection, caching, and dataset orchestration.
- `src/features`: point-in-time fundamental, analyst-history, price, market, and valuation features.
- `src/models`: linear/regularized, random-forest, histogram-gradient-boosting, and optional XGBoost comparisons.
- `src/backtest`: expanding-year splits, signal engine, transaction costs, metrics, and charts.
- `src/scanner`: current earnings discovery using `yfinance.Calendars.get_earnings_calendar()` and two-stage ranking.

## Windows PowerShell setup

Python 3.12 or newer is required.

```powershell
cd C:\path\to\earnings-quant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli status
```

Optional XGBoost:

```powershell
python -m pip install xgboost
```

Before SEC collection, replace the placeholder in `config.yaml` with a real identifying User-Agent containing your name or organization and contact email:

```yaml
sec:
  user_agent: "Your Name your.email@example.com"
```

The SEC asks automated clients to declare a User-Agent and currently caps automated access at 10 requests per second. The default project limit is a conservative 5 requests per second. Company Facts endpoints require no API key.

## Workflow

### Interactive terminal application

After collecting data, building the dataset, and training the models, open the keyboard-driven TUI:

```powershell
python -m src.cli tui
```

The application loads the upcoming earnings calendar. Use the arrow keys and `Enter` to select a company (or type its ticker), then choose **Analyze selected**. The forecast table compares the model with analyst consensus and shows:

- predicted EPS and year-over-year EPS growth;
- predicted revenue and year-over-year revenue growth;
- predicted operating margin and free cash flow;
- EPS/revenue differences versus analyst consensus;
- expected abnormal three-session return, probability of an up move, confidence, and signal.

Use `R` to refresh the calendar, `Ctrl+S` to analyze, and `Q` to quit. The TUI calls the same scanner and trained model bundle as the existing `scan` command; it does not retrain or change model logic while analyzing a ticker.

For first-time setup:

```powershell
python -m src.cli collect
python -m src.cli build-dataset
python -m src.cli train
python -m src.cli tui
```

For a quick development run:

```powershell
python -m src.cli collect --tickers AAPL MSFT NVDA AMZN GOOGL
python -m src.cli build-dataset
python -m src.cli status
```

The five-ticker development run is suitable for testing the pipeline, but it is intentionally too small to emit live `LONG` or `SHORT` signals. By default, the scanner requires a model trained on at least 20 distinct tickers. Run collection on the configured universe and retrain before using scanner signals.

For the configured deterministic 100-stock sample of the S&P 500 universe:

```powershell
python -m src.cli collect
python -m src.cli build-dataset
python -m src.cli train
python -m src.cli evaluate
python -m src.cli backtest
python -m src.cli scan --days 14 --top 20
```

Collection across 100 companies can take a while because requests are intentionally throttled and failures are isolated per ticker. Change `universe.target_size` to adjust the normal sample size, use `--limit 25` for an initial pipeline check, and rerun without `--refresh` to reuse the cache. An undersized cached fallback universe is refreshed automatically.

Other useful commands:

```powershell
python -m src.cli collect --limit 50
python -m src.cli collect --tickers AAPL MSFT --refresh
python -m src.cli scan --days 14 --top 5
python -m src.cli status
python -m pytest
```

## Data and artifacts

```text
data/raw/           normalized fundamentals, earnings, prices, metadata
data/processed/     one-row-per-earnings-event research dataset
data/models/        joblib bundle, features, parameters, dates, metrics
data/cache/         SEC JSON, universe cache, artifact metadata SQLite
results/            backtest rows and grouped statistics
results/charts/     cumulative return, drawdown, calibration, importance, hit rate, buckets
```

Historical fundamentals come from the SEC's `data.sec.gov/api/xbrl/companyfacts` endpoint. Concept aliases are normalized to a stable internal schema. Every fact is classified as `quarterly`, `annual`, `year_to_date`, or `other` using its filing form and duration; period start, duration, SEC frame, fiscal period, and form are retained. Year-to-date values are stored for auditability but never mixed into quarterly or annual model snapshots. Because Company Facts supplies a filing date rather than an exact dissemination time, the original `filed_date` is retained and `filed_at` conservatively makes the data available after that date ends, preventing a same-day earnings event from seeing its own filing. Instantaneous balance-sheet facts are aligned to the nearest statement period in the same filing. When a 10-K does not disclose standalone fourth-quarter income-statement values, Q4 additive values are derived as the annual total minus Q1–Q3. Free cash flow is operating cash flow minus absolute capital expenditure.

Yahoo/yfinance supplies adjusted and raw prices, event dates, historical EPS estimates/actuals where available, current estimates and revisions, metadata, and the upcoming US earnings calendar. Missing historical EPS estimates are recovered from reported EPS and Yahoo's surprise percentage only when the estimate is mathematically unambiguous. For annual event rows, the dataset uses a point-in-time proxy equal to the sum of the four quarterly EPS estimates available at those events; provenance and availability flags distinguish derived values from provider values. Yahoo's historical endpoint does not currently expose revenue consensus, so historical revenue-consensus features remain unavailable unless another provider supplies them; the pipeline accepts and tracks those fields without substituting reported revenue. Provider fields are optional: a missing estimate or failed ticker is logged and does not stop collection.

## Point-in-time methodology

Each historical row represents one company earnings event. The feature builder first determines whether the event corresponds to a quarterly statement or the annual statement following fiscal Q3. It then uses only matching-cadence history: quarterly features use standalone quarters and four-quarter trailing sums, while annual features use full-year history and the latest annual value. The statement cadence is also supplied to model training as a categorical feature. All timestamps are converted to UTC and a financial filing is admitted only when:

```text
filing filed_at < earnings event timestamp
```

Later amendments and future quarter values cannot enter the feature snapshot. Sparse comparative facts from later filings are consolidated column-by-column without erasing facts that were already public. The next SEC report may be used as a supervised label only when it is filed after the event and its period end is plausibly associated with that event. Label provenance is stored in `target_filed_at`; feature provenance is stored in `max_feature_filed_at`.

The dataset build and model training both call `assert_no_lookahead`. Tests deliberately add a future filing with an extreme value and verify that it cannot affect the event features. The same-day boundary is strict: a filing timestamp equal to the event timestamp is rejected.

Price features end strictly before the event timestamp. Post-event targets use close-to-close 1-, 3-, and 5-session stock returns minus matching SPY returns. After-market events skip the event-day close because it predates the announcement.

## Features and models

Fundamentals include quarterly/yearly growth, four-quarter trends, margins and margin changes, cash-flow growth, ROA/ROE, leverage, liquidity, dilution, asset growth, and revenue/EPS acceleration. Market features include 5/20/60/120-session momentum, volatility, volume, 52-week-high distance, SPY returns, and relative performance. Point-in-time valuation features include trailing/forward P/E, price/sales, EV/revenue, price/book, free-cash-flow yield, and expanding company-relative z-scores when enough observations exist.

Model A separately predicts actual EPS, revenue, operating margin, and free cash flow. Candidate validation includes linear regression, Ridge, Lasso, random forest, histogram gradient boosting, and XGBoost when installed. Model B predicts both `abnormal_return_3d` and `P(up)` using linear/logistic, regularized, forest, and boosting candidates.

Model B also includes mean-return and prior-probability baselines. A more complex reaction model is selected only when it beats the corresponding naive baseline on chronological validation data.

The reaction model is trained on expanding-window, out-of-fold Model A forecasts. It never receives Model A fitted values from the same observations. Model selection uses the last training year as validation; the configured `final_test_year` is excluded from training and remains untouched until `evaluate`.

Model metadata records feature names, selected estimators, training timestamp and period, candidate validation scores, and fit diagnostics. Models are serialized with joblib. Seeds are fixed in configuration.

## Walk-forward evaluation and backtest

`src/backtest/walk_forward.py` creates expanding annual folds such as:

```text
train 2016-2021, validate 2022, test 2023
train 2016-2022, validate 2023, test 2024
train 2016-2023, validate 2024, test 2025
```

No random shuffle is used. Default signals require both the predicted-return threshold and direction-consistent probability:

- `LONG`: predicted abnormal return above +2% and `P(up)` above minimum confidence.
- `SHORT`: predicted abnormal return below -2% and `P(down)` above minimum confidence.
- otherwise `NO_TRADE`.

The backtest applies commission plus entry and exit slippage. It reports signal counts, win rate, mean/median and cumulative returns, annualized return where meaningful, Sharpe, Sortino, maximum drawdown, profit factor, and directional accuracy. Breakdowns are produced for direction, sector, market-cap bucket, and confidence bucket.

This is an event-study backtest: signals can overlap, and the cumulative curve compounds event returns in timestamp order. It is not a capital-constrained portfolio simulator.

## Scanner output

The scanner fetches all qualifying calendar pages (Yahoo caps each page at 100), refreshes current quarterly and annual analyst data, infers the upcoming fiscal statement cadence from the latest visible filing, calculates matching point-in-time features, runs both models, and ranks by absolute expected abnormal return, confidence, then market cap. Output includes statement type/fiscal period, ticker, company, date/timing, sector, cadence-matched consensus and predicted EPS/revenue, predicted growth and surprises, predicted 3D return, probability up, confidence, and signal.

Financial level targets are anchored to each company's own prior statement: revenue and free cash flow are modeled as scale-relative values, while EPS is modeled as a change from prior EPS. Before a reaction forecast can become a signal, the scanner verifies training-universe coverage, cadence-specific history counts, revenue scale against both the latest report and analyst consensus, operating-margin bounds, and free-cash-flow scale. Failed validation produces `INSUFFICIENT_DATA`, explains the reason, and suppresses expected return and probability rather than emitting a misleading trade signal.

The scanner additionally requires saved walk-forward evidence for the reaction model. By default, overall ROC AUC must be at least `0.52` and reaction-return R² at least `0.0`; otherwise every live event is marked `INSUFFICIENT_DATA`. Retraining invalidates old evaluation artifacts, so `evaluate` must be run again before live signals can resume.

## Important limitations

- This is research software, not investment advice. There is no order execution or broker integration.
- The default universe is the **current** S&P 500, which introduces survivorship bias. For publication-quality work, provide a point-in-time constituent-history universe.
- Yahoo does not guarantee complete historical revenue consensus or estimate-revision snapshots. The project intentionally leaves unavailable historical fields missing; it never backfills today's analyst snapshot into old events. A licensed point-in-time estimates provider can be added behind the provider boundary.
- SEC XBRL tagging varies by filer. Alias normalization cannot recover facts a company did not tag comparably, and standalone fourth-quarter duration facts are often absent from Company Facts.
- Event timestamps and announced timing can be missing or revised. Ambiguous events use the normalized event date and should be reviewed before relying on results.
- Sector classification from current free metadata is not a full point-in-time classification history.
- Thresholds must be chosen on training/validation data, never by optimizing against final-test results.
- yfinance is intended for research/personal use and may change or throttle its unofficial Yahoo endpoints.

## Extending providers

Provider behavior is isolated under `src/providers`. A replacement should return the same normalized columns and preserve two timestamps: the fiscal period end and the time the information became public. When adding a historical analyst provider, store observation timestamps for every estimate and select only records observed before each event.

## Tests

```powershell
python -m pytest
```

The suite covers feature calculations, strict filing visibility, provenance auditing, two-stage model fit/prediction, chronological folds, signal thresholds, and trading costs. Add a regression test whenever a provider schema or timing rule changes.
