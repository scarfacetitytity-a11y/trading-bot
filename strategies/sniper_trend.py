"""Sniper Trend Strategy — for forex pairs and indices.

Built on the Sniper's entry logic but designed for trending markets.
Instead of exiting at the BB midline (mean-reversion), it rides the new
trend using a trailing ATR stop and a 3x ATR profit target.

Entry logic (ALL must be true):
  Long  : SMA50 > SMA200                        (uptrend confirmed)
           AND RSI < rsi_oversold               (deep pullback within trend)
           AND MACD bullish cross within 5 bars (momentum turning back up)
           AND price below BB lower band        (price stretched, entry at value)

  Short : SMA50 < SMA200                        (downtrend confirmed)
           AND RSI > rsi_overbought             (dead-cat bounce in downtrend)
           AND MACD bearish cross within 5 bars
           AND price above BB upper band

Exit (whichever hits first):
  1. Trailing ATR stop — starts at entry ± atr_mult*ATR, trails as price moves
  2. Profit target     — 3x ATR from entry (locked in, not trailing)
  3. Trend flip        — SMA50/200 crosses against position
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class SniperTrendStrategy(Strategy):
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
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 2.0,    # initial stop distance
        atr_target: float = 3.0,  # profit target distance
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
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.atr_target = atr_target
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"SniperTrend(sma={self.sma_fast}/{self.sma_slow}"
            f",rsi={self.rsi_oversold}/{self.rsi_overbought}"
            f",atr={self.atr_period}x{self.atr_mult}/tgt{self.atr_target})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # --- Trend filter (higher timeframe context) ---
        sma_fast  = close.rolling(self.sma_fast).mean()
        sma_slow  = close.rolling(self.sma_slow).mean()
        uptrend   = sma_fast > sma_slow
        downtrend = sma_fast < sma_slow

        # --- RSI ---
        rsi = _rsi(close, self.rsi_period)

        # --- MACD crossover ---
        ema_f       = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_s       = close.ewm(span=self.macd_slow,   adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        raw_bull    = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        raw_bear    = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        macd_bull   = raw_bull.rolling(5).max().astype(bool)
        macd_bear   = raw_bear.rolling(5).max().astype(bool)

        # --- Bollinger Bands ---
        bb_mid   = close.rolling(self.bb_period).mean()
        bb_std_s = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_s
        bb_lower = bb_mid - self.bb_std * bb_std_s

        # --- ATR ---
        atr = _atr(high, low, close, self.atr_period)

        warmup = max(self.sma_slow, self.macd_slow + self.macd_signal,
                     self.bb_period, self.atr_period)

        signals      = pd.Series(0, index=df.index)
        self._stops  = pd.Series(float("nan"), index=df.index)
        position     = 0
        stop_loss    = None
        take_profit  = None
        trail_stop   = None   # best trailing stop level reached so far

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = high.iloc[i]
            l       = low.iloc[i]
            atr_val = atr.iloc[i]

            # --- Trail the stop as price moves in our favour ---
            if position == 1:
                new_trail = h - self.atr_mult * atr_val
                if trail_stop is None or new_trail > trail_stop:
                    trail_stop = new_trail
                stop_loss = trail_stop

            elif position == -1:
                new_trail = l + self.atr_mult * atr_val
                if trail_stop is None or new_trail < trail_stop:
                    trail_stop = new_trail
                stop_loss = trail_stop

            # --- Stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None

            # --- Profit target hit ---
            if position == 1 and take_profit is not None and c >= take_profit:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None
            elif position == -1 and take_profit is not None and c <= take_profit:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None

            # --- Trend flip exit ---
            if position == 1 and downtrend.iloc[i]:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None
            elif position == -1 and uptrend.iloc[i]:
                position = 0; stop_loss = None; take_profit = None; trail_stop = None

            # --- Entries ---
            if position == 0 and not np.isnan(atr_val):
                long_entry = (
                    uptrend.iloc[i]
                    and rsi.iloc[i] < self.rsi_oversold
                    and macd_bull.iloc[i]
                    and c < bb_lower.iloc[i]
                )
                short_entry = (
                    not self.long_only
                    and downtrend.iloc[i]
                    and rsi.iloc[i] > self.rsi_overbought
                    and macd_bear.iloc[i]
                    and c > bb_upper.iloc[i]
                )

                if long_entry:
                    position    = 1
                    stop_loss   = c - self.atr_mult * atr_val
                    take_profit = c + self.atr_target * atr_val
                    trail_stop  = stop_loss
                elif short_entry:
                    position    = -1
                    stop_loss   = c + self.atr_mult * atr_val
                    take_profit = c - self.atr_target * atr_val
                    trail_stop  = stop_loss

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
