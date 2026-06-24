"""Run a backtest using synthetic OHLCV data (no MT5 needed)."""
import numpy as np
import pandas as pd

from backtests.engine import Backtest
from strategies.sniper import SniperStrategy
from strategies.rsi import RSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBands


def generate_synthetic_ohlcv(
    n_bars: int = 10_000,
    start: str = "2022-01-01",
    freq_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start=start, periods=n_bars, freq=f"{freq_minutes}min")

    # Geometric Brownian Motion with slight drift
    drift = 0.00001
    vol = 0.0008
    returns = rng.normal(drift, vol, n_bars)
    close = 1.2500 * np.exp(np.cumsum(returns))

    # Build OHLCV from close
    noise = rng.uniform(0.0001, 0.0005, n_bars)
    high = close + noise
    low = close - noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(100, 2000, n_bars)

    return pd.DataFrame({
        "time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
    })


if __name__ == "__main__":
    print("Generating synthetic GBPUSD M15 data (10,000 bars)...")
    df = generate_synthetic_ohlcv()

    strategies = [
        SniperStrategy(),
        RSIStrategy(),
        MACDStrategy(),
        BollingerBands(),
    ]

    for strat in strategies:
        bt = Backtest(strat, initial_capital=10_000, commission=0.0001)
        result = bt.run(df, symbol="GBPUSD_synthetic")
        result.print_summary()
