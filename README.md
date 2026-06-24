# Trading Bot (Forex/Indices) — Backtesting First

Status: project skeleton only. No strategy, data, or execution logic yet.

## Project layout

```
trading-bot/
├── data/
│   ├── raw/          # untouched historical exports from MT5 (gitignored)
│   └── processed/    # cleaned/validated data used by the backtester (gitignored)
├── strategies/        # strategy logic (signal generation)
├── backtests/         # backtesting engine + run scripts
├── execution/         # order execution / broker connectivity (MT5) - live phase
├── logs/              # runtime logs (gitignored)
├── config/
│   ├── config.example.yaml  # template - copy to config.yaml
│   └── config.yaml          # your local settings (gitignored, not yet created)
├── tests/             # unit/integration tests
├── .env.example       # template for MT5 credentials - copy to .env
├── requirements.txt
└── venv/              # Python virtual environment (gitignored)
```

## Setup

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create your local config from the templates
copy config\config.example.yaml config\config.yaml
copy .env.example .env
```

Then edit `config.yaml` (symbols, timeframe, MT5 terminal path) and `.env`
(MT5 login/password/server) with your real values. Neither file is committed
to version control.

## Running the data pipeline

```powershell
# Activate venv first, then:
python -m data.pipeline

# Re-process existing raw files without hitting MT5 again:
python -m data.pipeline --skip-fetch

# Validate processed output:
pytest
```

## Full comparison (all strategies × all symbols × all years)

```powershell
# Print ranked results table in the terminal:
python -m backtests.run_compare

# Single symbol only:
python -m backtests.run_compare --symbol XAUUSD

# Save CSV report + comparison charts:
python -m backtests.run_compare --save --charts
```

The output ranks every strategy per symbol per year by return, then prints a
"Best strategy" summary at the end showing the winner for each symbol/year.

## Running a single backtest

```powershell
# Run one strategy against all configured symbols:
python -m backtests.run_backtest --strategy sma_crossover
python -m backtests.run_backtest --strategy rsi
python -m backtests.run_backtest --strategy macd
python -m backtests.run_backtest --strategy bollinger_bands

# Run all strategies on one symbol and compare on a chart:
python -m backtests.run_backtest --strategy all --symbol XAUUSD --compare --chart

# Save charts to logs/reports/:
python -m backtests.run_backtest --strategy all --symbol US30 --compare --save-charts

# Custom capital / commission:
python -m backtests.run_backtest --capital 50000 --commission 0.00005
```

Available strategies: `sma_crossover`, `rsi`, `macd`, `bollinger_bands`

## Live trading

Make sure MT5 is open and logged into your account first.

```powershell
# Dry run — logs signals, places no real orders (always start here):
python -m execution.live_runner --dry-run

# Live on a single symbol:
python -m execution.live_runner --symbol XAUUSD --strategy rsi

# Live on all configured symbols simultaneously (one thread per symbol):
python -m execution.live_runner --strategy macd

# Stop: Ctrl+C — open positions are left as-is (bot doesn't force-close on exit)
```

> **Safety**: the bot blocks trading on real accounts unless you explicitly set
> `trading.allow_real_account: true` in `config.yaml`. Always test on demo first.

## Roadmap

1. [x] Project skeleton
2. [x] Historical data loader/validator (MT5 -> data/raw -> data/processed)
3. [x] Backtesting engine
4. [x] Strategies: SMA crossover, RSI, MACD, Bollinger Bands
5. [x] Performance reporting (equity curve + drawdown charts, comparison overlay)
6. [x] Live execution via MT5
