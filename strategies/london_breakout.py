"""London Breakout Strategy — improved.

Asian session (00:00-07:00 UTC) forms a consolidation range.
At London open (07:00 UTC) trade the breakout of that range.

Improvements over basic version:
  1. SMA 200 trend filter — only trade breakouts in direction of the trend
  2. Range-based stop     — stop at opposite side of Asian range (logical SL)
  3. Minimum range filter — skip days where Asian range < min_range_atr * ATR
                            (tight ranges = choppy = fake breakouts)

Entry:
  Long  : price > Asian high at 07:00 AND price > SMA200 AND range wide enough
  Short : price < Asian low  at 07:00 AND price < SMA200 AND range wide enough

Exit (whichever hits first):
  1. Range stop   — opposite Asian range extreme (Asian low for longs, high for shorts)
  2. ATR target   — atr_target * ATR from entry
  3. Session end  — force close at 17:00 UTC
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class LondonBreakoutStrategy(Strategy):
    def __init__(
        self,
        asian_start: int = 0,      # UTC hour Asian session starts
        asian_end: int = 7,        # UTC hour London opens
        session_close: int = 17,   # UTC hour to force-close all trades
        sma_period: int = 200,     # trend filter — only trade in SMA direction
        min_range_atr: float = 0.5,# Asian range must be >= this * ATR (avoids flat days)
        atr_period: int = 14,
        atr_target: float = 2.0,   # profit target in ATR multiples
        long_only: bool = False,
    ):
        self.asian_start = asian_start
        self.asian_end = asian_end
        self.session_close = session_close
        self.sma_period = sma_period
        self.min_range_atr = min_range_atr
        self.atr_period = atr_period
        self.atr_target = atr_target
        self.long_only = long_only

    @property
    def name(self) -> str:
        return (
            f"LondonBreakout(sma={self.sma_period}"
            f",minrange={self.min_range_atr}atr,tgt={self.atr_target}atr)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)

        sma    = close.rolling(self.sma_period).mean()
        atr    = _atr(high, low, close, self.atr_period)
        times  = pd.to_datetime(df["time"])
        hour   = times.dt.hour
        date   = times.dt.date

        warmup = max(self.sma_period, self.atr_period)

        # Pre-compute Asian range per day
        asian_mask = (hour >= self.asian_start) & (hour < self.asian_end)
        dates_list = list(date)
        asian_high = {}
        asian_low  = {}

        for idx, (d, is_asian, h_val, l_val) in enumerate(
            zip(dates_list, asian_mask, high, low)
        ):
            if is_asian:
                if d not in asian_high or h_val > asian_high[d]:
                    asian_high[d] = h_val
                if d not in asian_low or l_val < asian_low[d]:
                    asian_low[d] = l_val

        signals     = pd.Series(0, index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = hour.iloc[i]
            d       = dates_list[i]
            atr_val = atr.iloc[i]
            sma_val = sma.iloc[i]

            # --- Force close at session end ---
            if position != 0 and h >= self.session_close:
                position = 0; stop_loss = None; take_profit = None

            # --- Stop hit ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; take_profit = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; take_profit = None

            # --- Target hit ---
            if position == 1 and take_profit is not None and c >= take_profit:
                position = 0; stop_loss = None; take_profit = None
            elif position == -1 and take_profit is not None and c <= take_profit:
                position = 0; stop_loss = None; take_profit = None

            # --- London open: check for breakout ---
            if position == 0 and h == self.asian_end and not np.isnan(atr_val):
                a_high = asian_high.get(d)
                a_low  = asian_low.get(d)

                if a_high is not None and a_low is not None:
                    asian_range = a_high - a_low

                    # Skip tight/flat days
                    if asian_range < self.min_range_atr * atr_val:
                        signals.iloc[i] = position
                        continue

                    # Long: breakout above Asian high AND above SMA (uptrend)
                    if c > a_high and c > sma_val:
                        position    = 1
                        stop_loss   = a_low                        # stop below Asian range
                        take_profit = c + self.atr_target * atr_val

                    # Short: breakout below Asian low AND below SMA (downtrend)
                    elif not self.long_only and c < a_low and c < sma_val:
                        position    = -1
                        stop_loss   = a_high                       # stop above Asian range
                        take_profit = c - self.atr_target * atr_val

            signals.iloc[i] = position

        return signals


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
