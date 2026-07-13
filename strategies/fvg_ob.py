"""Fair Value Gap + Order Block strategy.

Setup:
  1. Detect 3-candle FVG: gap between candle[-2] and candle[0]
     Bull FVG: c[-2].high < c[0].low  (upward imbalance)
     Bear FVG: c[-2].low  > c[0].high (downward imbalance)
  2. Order Block filter: last opposing candle must exist within ob_lookback
     bars before FVG formation (confirms institutional origin of the move)
  3. Test: price retraces INTO the FVG zone (wick enters the gap)
  4. Entry: after test, close breaks back out in the FVG direction
     Bull FVG tested → close above FVG top  → Long
     Bear FVG tested → close below FVG bottom → Short
  5. Invalidation: close fully through the opposite FVG edge cancels the setup

Trade management:
  - Stop: beyond opposite FVG edge + ATR buffer
  - TP:   RR multiple from entry
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


class FVGOrderBlockStrategy(Strategy):
    def __init__(
        self,
        min_fvg_atr: float   = 0.1,   # minimum FVG size as fraction of ATR
        max_fvg_wait: int    = 50,     # bars to wait for price to test the FVG
        max_entry_wait: int  = 10,     # bars after test to wait for break confirmation
        rr_target: float     = 2.0,
        atr_period: int      = 14,
        atr_stop_buffer: float = 0.5,  # extra stop distance beyond FVG edge in ATR
        ob_lookback: int     = 30,     # bars to search for order block before FVG
        ob_required: bool    = True,   # require OB confirmation
        session_filter: bool = False,  # filter to London/NY sessions only
        max_active_fvgs: int = 5,      # cap on simultaneous tracked FVGs
        long_only: bool      = True,   # AiDEN constraint: no short positions
    ):
        self.min_fvg_atr     = min_fvg_atr
        self.max_fvg_wait    = max_fvg_wait
        self.max_entry_wait  = max_entry_wait
        self.rr_target       = rr_target
        self.atr_period      = atr_period
        self.atr_stop_buffer = atr_stop_buffer
        self.ob_lookback     = ob_lookback
        self.ob_required     = ob_required
        self.session_filter  = session_filter
        self.max_active_fvgs = max_active_fvgs
        self.long_only       = long_only

    @property
    def name(self) -> str:
        return (
            f"FVG_OB(gap={self.min_fvg_atr}atr"
            f",wait={self.max_fvg_wait}/{self.max_entry_wait}"
            f",rr={self.rr_target}"
            f",ob={'on' if self.ob_required else 'off'}"
            f",{'long_only' if self.long_only else 'bidir'})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        open_ = df["open"].astype(float)

        atr = _atr(high, low, close, self.atr_period)

        if self.session_filter:
            times      = pd.to_datetime(df["time"])
            hour       = times.dt.hour
            in_session = ((hour >= 7) & (hour < 9)) | ((hour >= 13) & (hour < 15))
        else:
            in_session = pd.Series(True, index=df.index)

        warmup = self.atr_period + 3

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)

        position    = 0
        stop_loss   = None
        take_profit = None

        # Each FVG: {dir, fvg_lo, fvg_hi, formed, tested, test_bar}
        active_fvgs: list[dict] = []

        for i in range(warmup, len(df)):
            cv      = close.iloc[i]
            lv      = low.iloc[i]
            hv      = high.iloc[i]
            atr_val = atr.iloc[i]

            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            # ── 1. Manage open position ────────────────────────────────────────
            if position == 1:
                if stop_loss is not None and lv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and hv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1 and not self.long_only:
                if stop_loss is not None and hv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and lv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            # ── 2. Detect new FVGs (requires 3 bars) ──────────────────────────
            if position == 0 and i >= warmup + 2:
                h2 = high.iloc[i - 2]
                l2 = low.iloc[i - 2]
                min_gap = self.min_fvg_atr * atr_val

                # Bullish FVG: gap between c[-2].high and c[0].low
                bull_gap = lv - h2
                if bull_gap >= min_gap:
                    if not self.ob_required or _has_ob(open_, close, i - 2, self.ob_lookback, "bull"):
                        active_fvgs.append({
                            "dir":      "bull",
                            "fvg_lo":   h2,     # bottom of the gap (c[-2].high)
                            "fvg_hi":   lv,     # top of the gap (c[0].low)
                            "formed":   i,
                            "tested":   False,
                            "test_bar": None,
                        })

                # Bearish FVG: gap between c[-2].low and c[0].high
                if not self.long_only:
                    bear_gap = l2 - hv
                    if bear_gap >= min_gap:
                        if not self.ob_required or _has_ob(open_, close, i - 2, self.ob_lookback, "bear"):
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   hv,     # bottom of the gap (c[0].high)
                                "fvg_hi":   l2,     # top of the gap (c[-2].low)
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                            })

                # Trim oldest if over cap
                if len(active_fvgs) > self.max_active_fvgs:
                    active_fvgs = active_fvgs[-self.max_active_fvgs:]

            # ── 3. Process active FVGs ─────────────────────────────────────────
            if position == 0:
                to_remove = []

                for fvg in active_fvgs:
                    fvg_lo = fvg["fvg_lo"]
                    fvg_hi = fvg["fvg_hi"]

                    if fvg["dir"] == "bull":
                        # Invalidated: close below zone bottom
                        if cv < fvg_lo:
                            to_remove.append(fvg)
                            continue

                        # Expire if never tested in time
                        if not fvg["tested"] and (i - fvg["formed"]) > self.max_fvg_wait:
                            to_remove.append(fvg)
                            continue

                        # Test: wick enters zone from above
                        if not fvg["tested"] and lv <= fvg_hi:
                            fvg["tested"]   = True
                            fvg["test_bar"] = i

                        # Entry: after test, close breaks above zone top
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv > fvg_hi and in_session.iloc[i]:
                                sl   = fvg_lo - self.atr_stop_buffer * atr_val
                                dist = cv - sl
                                if dist > 0:
                                    position    = 1
                                    stop_loss   = sl
                                    take_profit = cv + self.rr_target * dist
                                to_remove.append(fvg)

                    else:  # bear
                        # Invalidated: close above zone top
                        if cv > fvg_hi:
                            to_remove.append(fvg)
                            continue

                        # Expire if never tested in time
                        if not fvg["tested"] and (i - fvg["formed"]) > self.max_fvg_wait:
                            to_remove.append(fvg)
                            continue

                        # Test: wick enters zone from below
                        if not fvg["tested"] and hv >= fvg_lo:
                            fvg["tested"]   = True
                            fvg["test_bar"] = i

                        # Entry: after test, close breaks below zone bottom
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv < fvg_lo and in_session.iloc[i]:
                                sl   = fvg_hi + self.atr_stop_buffer * atr_val
                                dist = sl - cv
                                if dist > 0:
                                    position    = -1
                                    stop_loss   = sl
                                    take_profit = cv - self.rr_target * dist
                                to_remove.append(fvg)

                for fvg in to_remove:
                    if fvg in active_fvgs:
                        active_fvgs.remove(fvg)

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


def _has_ob(open_: pd.Series, close: pd.Series, start_i: int, lookback: int, fvg_dir: str) -> bool:
    """Check if there's a qualifying order block candle before the FVG.

    Bull FVG → needs a prior bearish candle (OB where buying accumulated).
    Bear FVG → needs a prior bullish candle (OB where selling accumulated).
    """
    end_i = max(0, start_i - lookback)
    for j in range(start_i, end_i, -1):
        if fvg_dir == "bull" and close.iloc[j] < open_.iloc[j]:
            return True
        if fvg_dir == "bear" and close.iloc[j] > open_.iloc[j]:
            return True
    return False
