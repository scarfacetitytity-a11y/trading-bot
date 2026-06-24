"""London Breakout Strategy.

Asian session (00:00-07:00 UTC) forms a consolidation range.
At London open (07:00 UTC) trade the breakout of that range.

Long  : price breaks ABOVE Asian session high at London open
Short : price breaks BELOW Asian session low at London open

Exit (whichever hits first):
  1. ATR stop    — entry ± atr_mult * ATR
  2. 2:1 target  — profit target = atr_target * ATR from entry
  3. Session end — close any open trade at 17:00 UTC (NY close)
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class LondonBreakoutStrategy(Strategy):
    def __init__(
        self,
        asian_start: int = 0,    # UTC hour Asian session starts
        asian_end: int = 7,      # UTC hour London opens
        session_close: int = 17, # UTC hour to force-close trades
        atr_period: int = 14,
        atr_mult: float = 1.5,   # stop distance
        atr_target: float = 3.0, # profit target distance
        long_only: bool = False,
    ):
        self.asian_start = asian_start
        self.asian_end = asian_end
        self.session_close = session_close
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.atr_target = atr_target
        self.long_only = long_only

    @property
    def name(self) -> str:
        return f"LondonBreakout(atr={self.atr_period}x{self.atr_mult}/tgt{self.atr_target})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)

        atr    = _atr(high, low, close, self.atr_period)
        times  = pd.to_datetime(df["time"])
        hour   = times.dt.hour
        date   = times.dt.date

        warmup = self.atr_period

        signals     = pd.Series(0, index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        # Pre-compute Asian range per day
        asian_mask  = (hour >= self.asian_start) & (hour < self.asian_end)
        asian_high  = {}
        asian_low   = {}
        for d in pd.Series(date).unique():
            mask = (pd.Series(date) == d) & asian_mask
            if mask.any():
                asian_high[d] = high[mask].max()
                asian_low[d]  = low[mask].min()

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = hour.iloc[i]
            d       = date[i]
            atr_val = atr.iloc[i]

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
                    if c > a_high:
                        position    = 1
                        stop_loss   = c - self.atr_mult * atr_val
                        take_profit = c + self.atr_target * atr_val
                    elif not self.long_only and c < a_low:
                        position    = -1
                        stop_loss   = c + self.atr_mult * atr_val
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
