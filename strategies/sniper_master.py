"""Sniper Master Strategy.

Combines the Sniper's high-precision confluence entries with a trend backbone,
ATR-based stop loss, and session filter.

Entry logic:
  Long  : SMA50 > SMA200 (uptrend)
           AND RSI < rsi_oversold (extreme pullback)
           AND MACD bullish cross within last 5 bars (momentum turning)
           AND price below BB lower band (stretched too far down)
           AND London or NY session (intraday only)

  Short : SMA50 < SMA200 (downtrend)
           AND RSI > rsi_overbought (extreme rally)
           AND MACD bearish cross within last 5 bars
           AND price above BB upper band (stretched too far up)
           AND session filter

Exit logic (whichever hits first):
  1. ATR stop   — price moves atr_mult * ATR against entry
  2. BB midline — price reverts to the BB middle band (take profit)
  3. Trend flip — SMA50/200 cross against position (cut and run)
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class SniperMasterStrategy(Strategy):
    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_cross_window: int = 5,    # bars to look back for MACD cross
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 1.5,         # stop = entry ± atr_mult * ATR
        session_filter: bool = True,
        long_only: bool = False,
    ):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.macd_cross_window = macd_cross_window
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.session_filter = session_filter
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"SniperMaster(sma={self.sma_fast}/{self.sma_slow}"
            f",rsi={self.rsi_oversold}/{self.rsi_overbought}"
            f",bb={self.bb_period},atr={self.atr_period}x{self.atr_mult})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # --- Trend ---
        sma_fast  = close.rolling(self.sma_fast).mean()
        sma_slow  = close.rolling(self.sma_slow).mean()
        uptrend   = sma_fast > sma_slow
        downtrend = sma_fast < sma_slow

        # --- RSI ---
        rsi = _rsi(close, self.rsi_period)

        # --- MACD crossover (recent within window) ---
        ema_f       = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_s       = close.ewm(span=self.macd_slow,   adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        bull_cross  = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        bear_cross  = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        recent_bull = bull_cross.rolling(self.macd_cross_window).max().astype(bool)
        recent_bear = bear_cross.rolling(self.macd_cross_window).max().astype(bool)

        # --- Bollinger Bands ---
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s

        # --- ATR ---
        atr = _atr(high, low, close, self.atr_period)

        # --- Session filter (skip on daily+ bars) ---
        if self.session_filter and "time" in df.columns:
            times = pd.to_datetime(df["time"])
            median_hours = times.diff().dt.total_seconds().median() / 3600
            if median_hours >= 20:
                in_session = pd.Series(True, index=df.index)
            else:
                hour = times.dt.hour
                in_session = ((hour >= 7) & (hour < 12)) | ((hour >= 13) & (hour < 17))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.sma_slow, self.macd_slow + self.macd_signal,
                     self.bb_period, self.atr_period)

        signals   = pd.Series(0, index=df.index)
        position  = 0
        stop_loss = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            atr_val = atr.iloc[i]

            # --- ATR stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0
                stop_loss = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0
                stop_loss = None

            # --- BB midline take profit ---
            if position == 1 and c >= bb_mid.iloc[i]:
                position = 0
                stop_loss = None
            elif position == -1 and c <= bb_mid.iloc[i]:
                position = 0
                stop_loss = None

            # --- Trend flip exit ---
            if position == 1 and downtrend.iloc[i]:
                position = 0
                stop_loss = None
            elif position == -1 and uptrend.iloc[i]:
                position = 0
                stop_loss = None

            # --- Entries (only when flat) ---
            if position == 0 and not np.isnan(atr_val):
                long_entry = (
                    uptrend.iloc[i]
                    and rsi.iloc[i] < self.rsi_oversold
                    and recent_bull.iloc[i]
                    and c < bb_lower.iloc[i]
                    and in_session.iloc[i]
                )
                short_entry = (
                    not self.long_only
                    and downtrend.iloc[i]
                    and rsi.iloc[i] > self.rsi_overbought
                    and recent_bear.iloc[i]
                    and c > bb_upper.iloc[i]
                    and in_session.iloc[i]
                )

                if long_entry:
                    position  = 1
                    stop_loss = c - self.atr_mult * atr_val
                elif short_entry:
                    position  = -1
                    stop_loss = c + self.atr_mult * atr_val

            signals.iloc[i] = position

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
