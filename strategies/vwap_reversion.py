"""VWAP Reversion strategy.

Concept: VWAP resets each calendar day and acts as the session's fair value
anchor. Institutions use it to benchmark executions. Price that strays far
from VWAP tends to revert — especially inside a trending session.

Entry logic:
  Long:  price pulled back to within vwap_band of VWAP AND close > VWAP AND
         close > EMA (trend filter, optional) AND RSI not overbought
  Short: price stretched above VWAP band AND close < VWAP AND
         close < EMA (trend filter) AND RSI not oversold

  With trend_ema_period=0 the EMA trend filter is disabled.

Stop: ATR-based below recent swing low / above swing high.
TP:   entry + rr_target × risk (from stop).

Note on VWAP on H1 vs M5:
  On H1 data volume is less meaningful but VWAP still identifies the
  session price anchor. On M5 it's more precise. Both work.
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP that resets each calendar day.

    Falls back to cumulative price × volume when 'volume' column is absent
    or all zeros (common with synthetic/broker data for metals/indices).
    """
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    tp     = (high + low + close) / 3   # typical price

    has_vol = "tick_volume" in df.columns or "volume" in df.columns
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"

    if has_vol:
        vol = df[vol_col].astype(float)
        if vol.sum() == 0:
            has_vol = False

    dates = pd.to_datetime(df["time"]).dt.date

    vwap = pd.Series(float("nan"), index=df.index)
    for day, grp in df.groupby(dates):
        idx = grp.index
        tp_day = tp.loc[idx]
        if has_vol:
            v_day  = vol.loc[idx]
            cum_tv = (tp_day * v_day).cumsum()
            cum_v  = v_day.cumsum().replace(0, float("nan"))
            vwap.loc[idx] = cum_tv / cum_v
        else:
            # Equal-weight VWAP (TWAP) when no volume
            vwap.loc[idx] = tp_day.expanding().mean()

    return vwap


class VWAPReversionStrategy(Strategy):
    """Revert to daily VWAP after price deviates by > vwap_atr_band ATR units."""

    def __init__(
        self,
        vwap_atr_band: float   = 1.0,    # price is "far" when > this many ATR from VWAP
        atr_period: int        = 14,
        rr_target: float       = 2.0,
        atr_stop_mult: float   = 1.5,    # stop = entry ± this × ATR
        trend_ema_period: int  = 0,      # 0 = disabled; 200 = only trade with EMA trend
        rsi_period: int        = 14,
        rsi_ob: float          = 70.0,   # skip long when RSI > this
        rsi_os: float          = 30.0,   # skip short when RSI < this
        session_filter: bool   = True,
        long_only: bool        = False,
    ):
        self.vwap_atr_band    = vwap_atr_band
        self.atr_period       = atr_period
        self.rr_target        = rr_target
        self.atr_stop_mult    = atr_stop_mult
        self.trend_ema_period = trend_ema_period
        self.rsi_period       = rsi_period
        self.rsi_ob           = rsi_ob
        self.rsi_os           = rsi_os
        self.session_filter   = session_filter
        self.long_only        = long_only

    @property
    def name(self) -> str:
        return (
            f"VWAP_Rev(band={self.vwap_atr_band}atr"
            f",ema={self.trend_ema_period}"
            f",rr={self.rr_target}"
            f",{'long_only' if self.long_only else 'bidir'})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        atr  = _atr(high, low, close, self.atr_period)
        vwap = _daily_vwap(df)

        # RSI
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss   = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs     = gain / loss.replace(0, float("nan"))
        rsi    = 100 - 100 / (1 + rs)

        # Trend EMA filter
        if self.trend_ema_period > 0:
            ema_trend = close.ewm(span=self.trend_ema_period, adjust=False).mean()
            bull_trend = close > ema_trend
            bear_trend = close < ema_trend
        else:
            bull_trend = pd.Series(True, index=df.index)
            bear_trend = pd.Series(True, index=df.index)

        if self.session_filter:
            times      = pd.to_datetime(df["time"])
            hour       = times.dt.hour
            in_session = ((hour >= 7) & (hour < 9)) | ((hour >= 13) & (hour < 15))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.atr_period, self.rsi_period,
                     self.trend_ema_period if self.trend_ema_period else 0) + 5

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        for i in range(warmup, len(df)):
            cv      = close.iloc[i]
            lv      = low.iloc[i]
            hv      = high.iloc[i]
            atr_val = atr.iloc[i]
            vwap_v  = vwap.iloc[i]
            rsi_v   = rsi.iloc[i]

            if np.isnan(atr_val) or np.isnan(vwap_v) or np.isnan(rsi_v):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            band = self.vwap_atr_band * atr_val

            # Manage open position
            if position == 1:
                if stop_loss is not None and lv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and hv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and hv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and lv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            # Entry: price reverts back toward VWAP after stretching away
            if position == 0 and in_session.iloc[i]:
                price_above_vwap = cv > vwap_v
                price_below_vwap = cv < vwap_v
                dist_from_vwap   = abs(cv - vwap_v)

                # Long: price pulled back close to VWAP from below (was below,
                # now near VWAP) — OR price pierces VWAP upward from a deviation
                long_entry = (
                    price_above_vwap
                    and dist_from_vwap <= band          # near VWAP (not over-extended)
                    and bull_trend.iloc[i]
                    and rsi_v < self.rsi_ob
                )

                short_entry = (
                    not self.long_only
                    and price_below_vwap
                    and dist_from_vwap <= band
                    and bear_trend.iloc[i]
                    and rsi_v > self.rsi_os
                )

                if long_entry:
                    sl   = cv - self.atr_stop_mult * atr_val
                    dist = cv - sl
                    if dist > 0:
                        position    = 1
                        stop_loss   = sl
                        take_profit = cv + self.rr_target * dist

                elif short_entry:
                    sl   = cv + self.atr_stop_mult * atr_val
                    dist = sl - cv
                    if dist > 0:
                        position    = -1
                        stop_loss   = sl
                        take_profit = cv - self.rr_target * dist

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


class VWAPMomentumStrategy(Strategy):
    """Trade momentum when price breaks decisively through VWAP.

    Entry: close crosses VWAP with strong body (>body_pct of range) AND
           price was on wrong side of VWAP for at least flip_lookback bars.
    """

    def __init__(
        self,
        body_pct: float       = 0.6,
        flip_lookback: int    = 3,
        atr_period: int       = 14,
        atr_stop_mult: float  = 1.5,
        rr_target: float      = 2.0,
        session_filter: bool  = True,
        long_only: bool       = False,
    ):
        self.body_pct      = body_pct
        self.flip_lookback = flip_lookback
        self.atr_period    = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.rr_target     = rr_target
        self.session_filter = session_filter
        self.long_only     = long_only

    @property
    def name(self) -> str:
        return (
            f"VWAP_Mom(body={self.body_pct}"
            f",flip={self.flip_lookback}"
            f",rr={self.rr_target}"
            f",{'long_only' if self.long_only else 'bidir'})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        open_ = df["open"].astype(float)

        atr  = _atr(high, low, close, self.atr_period)
        vwap = _daily_vwap(df)

        above_vwap = close > vwap
        body        = (close - open_).abs()
        candle_range = (high - low).replace(0, float("nan"))
        strong_body  = body >= self.body_pct * candle_range

        if self.session_filter:
            times      = pd.to_datetime(df["time"])
            hour       = times.dt.hour
            in_session = ((hour >= 7) & (hour < 9)) | ((hour >= 13) & (hour < 15))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = max(self.atr_period, self.flip_lookback) + 5

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        for i in range(warmup, len(df)):
            cv      = close.iloc[i]
            lv      = low.iloc[i]
            hv      = high.iloc[i]
            atr_val = atr.iloc[i]
            vwap_v  = vwap.iloc[i]

            if np.isnan(atr_val) or np.isnan(vwap_v):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss is not None and lv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and hv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and hv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and lv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and in_session.iloc[i] and strong_body.iloc[i]:
                # Was below VWAP for flip_lookback bars, now closes above = bullish flip
                was_below = all(not above_vwap.iloc[i - j] for j in range(1, self.flip_lookback + 1))
                was_above = all(above_vwap.iloc[i - j] for j in range(1, self.flip_lookback + 1))

                if above_vwap.iloc[i] and was_below and close.iloc[i] > open_.iloc[i]:
                    sl   = lv - 0.2 * atr_val
                    dist = cv - sl
                    if dist > 0:
                        position    = 1
                        stop_loss   = sl
                        take_profit = cv + self.rr_target * dist

                elif not self.long_only and not above_vwap.iloc[i] and was_above and close.iloc[i] < open_.iloc[i]:
                    sl   = hv + 0.2 * atr_val
                    dist = sl - cv
                    if dist > 0:
                        position    = -1
                        stop_loss   = sl
                        take_profit = cv - self.rr_target * dist

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals
