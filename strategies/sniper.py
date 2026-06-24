"""Forex Sniper strategy.

Only enters when RSI, MACD, and Bollinger Bands all confluence:
  Long:  RSI oversold  + MACD line crossing above signal + price below BB lower
  Short: RSI overbought + MACD line crossing below signal + price above BB upper
  Exit:  price reverts to BB middle band
"""
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
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"Sniper(rsi={self.rsi_period},{self.rsi_oversold}/{self.rsi_overbought}"
            f",macd={self.macd_fast}/{self.macd_slow}/{self.macd_signal}"
            f",bb={self.bb_period},{self.bb_std})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)

        rsi = _rsi(close, self.rsi_period)

        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        raw_bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        raw_bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        # True if a crossover happened within the last 5 bars
        macd_bullish_cross = raw_bullish_cross.rolling(5).max().astype(bool)
        macd_bearish_cross = raw_bearish_cross.rolling(5).max().astype(bool)

        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std

        warmup = max(self.rsi_period, self.macd_slow + self.macd_signal, self.bb_period)

        signals = pd.Series(0, index=df.index)
        position = 0

        for i in range(warmup, len(df)):
            c = close.iloc[i]

            long_entry = (
                rsi.iloc[i] < self.rsi_oversold
                and macd_bullish_cross.iloc[i]
                and c < bb_lower.iloc[i]
            )
            short_entry = (
                not self.long_only
                and rsi.iloc[i] > self.rsi_overbought
                and macd_bearish_cross.iloc[i]
                and c > bb_upper.iloc[i]
            )
            long_exit = position == 1 and c >= bb_mid.iloc[i]
            short_exit = position == -1 and c <= bb_mid.iloc[i]

            if long_entry:
                position = 1
            elif short_entry:
                position = -1
            elif long_exit or short_exit:
                position = 0

            signals.iloc[i] = position

        return signals


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
