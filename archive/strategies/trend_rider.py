"""TrendRider strategy — for forex pairs and indices (non-gold).

Logic:
  1. SMA 200   — defines trend direction (above = uptrend, below = downtrend)
  2. ADX > 20  — confirms the market is actually trending, not ranging
  3. RSI 35-55 — price has pulled back inside the trend (not chasing a spike)
  4. MACD cross — momentum resuming in trend direction (entry trigger)

  Long:  close > SMA200 AND ADX > 20 AND RSI in [35, 55] AND MACD bullish cross
  Short: close < SMA200 AND ADX > 20 AND RSI in [45, 65] AND MACD bearish cross

Exit (whichever hits first):
  1. Trailing ATR stop — trails price to lock in profits as trend extends
  2. Initial ATR stop  — hard floor if trade moves immediately against us
  3. Session end       — force close at session_end hour (H1 only)
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class TrendRiderStrategy(Strategy):
    def __init__(
        self,
        sma_period: int = 200,
        rsi_period: int = 14,
        rsi_pull_lo: float = 35.0,   # pullback zone lower bound (longs)
        rsi_pull_hi: float = 55.0,   # pullback zone upper bound (longs)
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,  # initial hard stop
        atr_trail_mult: float = 2.5, # trailing stop distance
        adx_period: int = 14,
        adx_min: float = 25.0,       # skip ranging markets below this ADX
        session_start: int = 7,      # UTC hour — only trade inside session
        session_end: int = 17,       # UTC hour — force close at session end
        long_only: bool = False,
    ):
        self.sma_period    = sma_period
        self.rsi_period    = rsi_period
        self.rsi_pull_lo   = rsi_pull_lo
        self.rsi_pull_hi   = rsi_pull_hi
        self.macd_fast     = macd_fast
        self.macd_slow     = macd_slow
        self.macd_signal   = macd_signal
        self.atr_period    = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult
        self.adx_period    = adx_period
        self.adx_min       = adx_min
        self.session_start = session_start
        self.session_end   = session_end
        self.long_only     = long_only

    @property
    def name(self) -> str:
        return (
            f"TrendRider(sma={self.sma_period},adx={self.adx_min}"
            f",rsi={self.rsi_pull_lo}/{self.rsi_pull_hi}"
            f",trail={self.atr_trail_mult}atr)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # Detect daily bars — skip session filter if median gap >= 20 hours
        times = pd.to_datetime(df["time"])
        median_gap_hours = times.diff().dt.total_seconds().median() / 3600
        use_session = median_gap_hours < 20

        hour = times.dt.hour if use_session else pd.Series(self.session_start, index=df.index)

        sma = close.rolling(self.sma_period).mean()
        rsi = _rsi(close, self.rsi_period)
        atr = _atr(high, low, close, self.atr_period)
        adx = _adx(high, low, close, self.adx_period)

        ema_fast    = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=self.macd_slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        raw_bull    = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        raw_bear    = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        macd_bull   = raw_bull.rolling(5).max().astype(bool)
        macd_bear   = raw_bear.rolling(5).max().astype(bool)

        warmup = max(self.sma_period, self.adx_period * 2,
                     self.macd_slow + self.macd_signal, self.rsi_period, self.atr_period)

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        trail_stop  = None

        for i in range(warmup, len(df)):
            c       = close.iloc[i]
            h       = high.iloc[i]
            l       = low.iloc[i]
            atr_val = atr.iloc[i]
            hr      = hour.iloc[i]

            if np.isnan(atr_val) or np.isnan(adx.iloc[i]):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            in_session = self.session_start <= hr < self.session_end

            # --- Force close at session end ---
            if position != 0 and use_session and hr >= self.session_end:
                position = 0; stop_loss = None; trail_stop = None

            # --- Hard ATR stop ---
            if position == 1 and stop_loss is not None and c <= stop_loss:
                position = 0; stop_loss = None; trail_stop = None
            elif position == -1 and stop_loss is not None and c >= stop_loss:
                position = 0; stop_loss = None; trail_stop = None

            # --- Trailing stop: ratchet in direction of trade ---
            if position == 1 and trail_stop is not None:
                new_trail = c - self.atr_trail_mult * atr_val
                trail_stop = max(trail_stop, new_trail)
                if c <= trail_stop:
                    position = 0; stop_loss = None; trail_stop = None

            elif position == -1 and trail_stop is not None:
                new_trail = c + self.atr_trail_mult * atr_val
                trail_stop = min(trail_stop, new_trail)
                if c >= trail_stop:
                    position = 0; stop_loss = None; trail_stop = None

            # --- Entries: only during session, only in trending markets ---
            if position == 0 and in_session and not np.isnan(atr_val):
                trending  = adx.iloc[i] >= self.adx_min
                rsi_val   = rsi.iloc[i]

                # Long: uptrend + RSI pullback zone + MACD resuming up
                if (c > sma.iloc[i]
                        and trending
                        and self.rsi_pull_lo <= rsi_val <= self.rsi_pull_hi
                        and macd_bull.iloc[i]):
                    position   = 1
                    stop_loss  = c - self.atr_stop_mult * atr_val
                    trail_stop = c - self.atr_trail_mult * atr_val

                # Short: downtrend + RSI in upper pullback zone + MACD resuming down
                elif (not self.long_only
                        and c < sma.iloc[i]
                        and trending
                        and (100 - self.rsi_pull_hi) <= rsi_val <= (100 - self.rsi_pull_lo)
                        and macd_bear.iloc[i]):
                    position   = -1
                    stop_loss  = c + self.atr_stop_mult * atr_val
                    trail_stop = c + self.atr_trail_mult * atr_val

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


# ── Indicators ────────────────────────────────────────────────────────────────

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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average Directional Index — measures trend strength (direction-agnostic)."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # Where both are positive, keep only the larger one
    mask_both = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[mask_both & (minus_dm >= plus_dm)]  = 0
    minus_dm[mask_both & (plus_dm > minus_dm)]  = 0

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_s    = tr.rolling(period).mean()
    plus_di  = 100 * plus_dm.rolling(period).mean()  / atr_s.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_s.replace(0, float("nan"))

    dx  = (100 * (plus_di - minus_di).abs()
           / (plus_di + minus_di).replace(0, float("nan")))
    return dx.rolling(period).mean()
