"""Shared technical indicators — single source of truth.

Centralises the ATR / RSI helpers that were duplicated across every strategy, and
is the place to add new ones as our knowledge grows (VWAP added; extend here —
do NOT redefine indicators per-strategy).

Import as, e.g.:  from strategies.indicators import atr as _atr, rsi as _rsi, session_vwap
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (Wilder TR, simple rolling mean)."""
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (rolling-mean gains/losses)."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP over the whole series (typical price × volume)."""
    tp = (high + low + close) / 3.0
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def session_vwap(df: pd.DataFrame, vol_col: str = "tick_volume") -> pd.Series:
    """VWAP that RESETS each UTC day — the intraday reference price institutions
    trade around. df needs columns: time, high, low, close, <vol_col>."""
    day = pd.to_datetime(df["time"]).dt.normalize()
    tp  = (df["high"] + df["low"] + df["close"]) / 3.0
    pv  = (tp * df[vol_col]).groupby(day).cumsum()
    vv  = df[vol_col].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper, mid, lower, bandwidth) where bandwidth = (upper - lower) / mid.
    Bandwidth is normalised so squeeze thresholds are price-independent.
    """
    mid       = close.rolling(period).mean()
    sigma     = close.rolling(period).std()
    upper     = mid + std_dev * sigma
    lower     = mid - std_dev * sigma
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, bandwidth
