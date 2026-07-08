"""Forex Sniper strategy — XAUUSD H1.

Entry confluence (all must be true on the same bar):
  Long:  RSI oversold  + MACD bullish cross (within 5 bars) + price below BB lower
  Short: RSI overbought + MACD bearish cross (within 5 bars) + price above BB upper

Exit (whichever hits first):
  1. ATR stop  — price moves atr_mult * ATR against entry (cuts losers fast)
  2. BB middle — price reverts to BB midline (take profit)

Best config on XAUUSD H1 2015-2025: RSI 35/65, +107.9%, 59.4% WR, 224 trades, Sharpe 1.705
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class SniperStrategy(Strategy):
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        long_only: bool = False,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"Sniper(rsi={self.rsi_oversold}/{self.rsi_overbought}"
            f",bb={self.bb_period},atr={self.atr_period}x{self.atr_mult})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        rsi = _rsi(close, self.rsi_period)

        ema_fast    = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=self.macd_slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        raw_bull    = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        raw_bear    = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        macd_bull   = raw_bull.rolling(5).max().astype(bool)
        macd_bear   = raw_bear.rolling(5).max().astype(bool)

        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s

        atr = _atr(high, low, close, self.atr_period)

        warmup = max(self.rsi_period, self.macd_slow + self.macd_signal,
                     self.bb_period, self.atr_period)

        signals        = pd.Series(0, index=df.index)
        self._stops    = pd.Series(float("nan"), index=df.index)
        position  = 0
        stop_loss = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            atr_val = atr.iloc[i]

            # --- ATR stop ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None

            # --- BB midline take profit ---
            if position == 1 and c >= bb_mid.iloc[i]:
                position = 0; stop_loss = None
            elif position == -1 and c <= bb_mid.iloc[i]:
                position = 0; stop_loss = None

            # --- Entries ---
            if position == 0 and not np.isnan(atr_val):
                if (rsi.iloc[i] < self.rsi_oversold
                        and macd_bull.iloc[i]
                        and c < bb_lower.iloc[i]):
                    position  = 1
                    stop_loss = c - self.atr_mult * atr_val

                elif (not self.long_only
                        and rsi.iloc[i] > self.rsi_overbought
                        and macd_bear.iloc[i]
                        and c > bb_upper.iloc[i]):
                    position  = -1
                    stop_loss = c + self.atr_mult * atr_val

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
