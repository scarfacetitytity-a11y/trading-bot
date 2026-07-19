"""ICT AMD — Accumulation, Manipulation, Displacement strategies.

Multi-timeframe liquidity detection: sweeps are identified at multiple
swing_lookbacks simultaneously (e.g. 48/144/288 bars = 4h/12h/24h on M5),
catching intraday AMD setups as well as larger session-level ones.
When multiple windows fire on the same bar the smallest (tightest) level
is used for zone/stop placement.

Two entry versions:
  ICTAMDDisplacementStrategy  — enter at close of displacement candle
  ICTAMDBreakerStrategy       — enter on pullback into breaker zone
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy


class ICTAMDBase(Strategy):
    def __init__(
        self,
        swing_lookbacks: list  = None,   # bars; default [48, 144, 288] = 4h/12h/24h on M5
        sweep_window: int      = 3,
        sweep_min_atr: float   = 0.3,
        body_pct: float        = 0.55,
        atr_period: int        = 14,
        atr_stop_buffer: float = 0.2,
        rr_target: float       = 2.0,
        session_filter: bool   = True,
        htf_sma_period: int    = 0,
        eq_persist: int        = 0,
        long_only: bool        = False,
    ):
        self.swing_lookbacks   = sorted(swing_lookbacks or [48, 144, 288])
        self.sweep_window      = sweep_window
        self.sweep_min_atr     = sweep_min_atr
        self.body_pct          = body_pct
        self.atr_period        = atr_period
        self.atr_stop_buffer   = atr_stop_buffer
        self.rr_target         = rr_target
        self.session_filter    = session_filter
        self.htf_sma_period    = htf_sma_period
        self.eq_persist        = eq_persist
        self.long_only         = long_only

    def _compute(self, df: pd.DataFrame):
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        open_ = df["open"].astype(float)

        atr = _atr(high, low, close, self.atr_period)

        if self.htf_sma_period > 0:
            htf_sma        = close.rolling(self.htf_sma_period).mean()
            slope_lookback = max(1, self.htf_sma_period // 3)
            htf_bull       = htf_sma > htf_sma.shift(slope_lookback)
            htf_bear       = htf_sma < htf_sma.shift(slope_lookback)
        else:
            htf_bull = pd.Series(True, index=df.index)
            htf_bear = pd.Series(True, index=df.index)

        # Per-lookback sweep detection — smallest lookback first
        per_lookback = []
        bull_sweep_recent_any = pd.Series(False, index=df.index)
        bear_sweep_recent_any = pd.Series(False, index=df.index)

        for lb in self.swing_lookbacks:
            swing_low  = low.shift(1).rolling(lb).min()
            swing_high = high.shift(1).rolling(lb).max()

            bull_wick = swing_low - low
            bear_wick = high - swing_high

            bull_sweep = (
                (low < swing_low)
                & (close > swing_low)
                & (bull_wick >= self.sweep_min_atr * atr)
                & htf_bull
            )
            bear_sweep = (
                (high > swing_high)
                & (close < swing_high)
                & (bear_wick >= self.sweep_min_atr * atr)
                & htf_bear
            )

            if self.eq_persist > 0:
                bull_sweep = bull_sweep & (swing_low  == swing_low.shift(self.eq_persist))
                bear_sweep = bear_sweep & (swing_high == swing_high.shift(self.eq_persist))

            bull_recent = bull_sweep.rolling(self.sweep_window).max().astype(bool)
            bear_recent = bear_sweep.rolling(self.sweep_window).max().astype(bool)

            per_lookback.append(dict(
                lb=lb,
                bull_sweep=bull_sweep,
                bear_sweep=bear_sweep,
                bull_recent=bull_recent,
                bear_recent=bear_recent,
                swing_low=swing_low,
                swing_high=swing_high,
            ))
            bull_sweep_recent_any = bull_sweep_recent_any | bull_recent
            bear_sweep_recent_any = bear_sweep_recent_any | bear_recent

        body         = (close - open_).abs()
        candle_range = (high - low).replace(0, float("nan"))
        strong_body  = body >= self.body_pct * candle_range
        bull_disp    = strong_body & (close > open_)
        bear_disp    = strong_body & (close < open_)

        times      = pd.to_datetime(df["time"])
        hour       = times.dt.hour
        in_session = ((hour >= 7) & (hour < 9)) | ((hour >= 13) & (hour < 15))
        if not self.session_filter:
            in_session = pd.Series(True, index=df.index)

        warmup = max(
            max(self.swing_lookbacks) + self.eq_persist + self.sweep_window,
            self.atr_period,
            self.htf_sma_period if self.htf_sma_period > 0 else 0,
        ) + 1

        return dict(
            close=close, high=high, low=low, open_=open_, atr=atr,
            per_lookback=per_lookback,
            bull_sweep_recent_any=bull_sweep_recent_any,
            bear_sweep_recent_any=bear_sweep_recent_any,
            bull_disp=bull_disp, bear_disp=bear_disp,
            in_session=in_session,
            warmup=warmup,
        )


def _tightest_bull_level(per_lookback, i):
    """Return swing_low from the smallest lookback with an active recent bull sweep."""
    for ld in per_lookback:
        if ld["bull_recent"].iloc[i]:
            v = ld["swing_low"].iloc[i]
            if not np.isnan(v):
                return v
    return float("nan")


def _tightest_bear_level(per_lookback, i):
    for ld in per_lookback:
        if ld["bear_recent"].iloc[i]:
            v = ld["swing_high"].iloc[i]
            if not np.isnan(v):
                return v
    return float("nan")


def _immediate_bull_level(per_lookback, i):
    """Return swing_low from smallest lookback where bull_sweep fired THIS bar."""
    for ld in per_lookback:
        if ld["bull_sweep"].iloc[i] or (i > 0 and ld["bull_sweep"].iloc[i - 1]):
            v = ld["swing_low"].iloc[i]
            if not np.isnan(v):
                return v
    return float("nan")


def _immediate_bear_level(per_lookback, i):
    for ld in per_lookback:
        if ld["bear_sweep"].iloc[i] or (i > 0 and ld["bear_sweep"].iloc[i - 1]):
            v = ld["swing_high"].iloc[i]
            if not np.isnan(v):
                return v
    return float("nan")


# ── Version 1 — Displacement entry ───────────────────────────────────────────

class ICTAMDDisplacementStrategy(ICTAMDBase):
    """Enter at close of displacement candle after any lookback detects a sweep."""

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return f"ICTAMDDisplacement(swing=[{lbs}],eq={self.eq_persist},htf={self.htf_sma_period},rr={self.rr_target})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close         = c["close"]
        atr           = c["atr"]
        per_lookback  = c["per_lookback"]
        bull_recent   = c["bull_sweep_recent_any"]
        bear_recent   = c["bear_sweep_recent_any"]
        bull_disp     = c["bull_disp"]
        bear_disp     = c["bear_disp"]
        in_session    = c["in_session"]
        warmup        = c["warmup"]

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        for i in range(warmup, len(df)):
            cv      = close.iloc[i]
            atr_val = atr.iloc[i]

            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss   is not None and cv <= stop_loss:
                    position = 0; stop_loss = None; take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = None; take_profit = None
            elif position == -1:
                if stop_loss   is not None and cv >= stop_loss:
                    position = 0; stop_loss = None; take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = None; take_profit = None

            if position == 0 and in_session.iloc[i]:
                if bull_recent.iloc[i] and bull_disp.iloc[i]:
                    sl_level = _tightest_bull_level(per_lookback, i)
                    if not np.isnan(sl_level):
                        sl   = sl_level - self.atr_stop_buffer * atr_val
                        dist = cv - sl
                        if dist > 0:
                            position    = 1
                            stop_loss   = sl
                            take_profit = cv + self.rr_target * dist

                elif not self.long_only and bear_recent.iloc[i] and bear_disp.iloc[i]:
                    sh_level = _tightest_bear_level(per_lookback, i)
                    if not np.isnan(sh_level):
                        sl   = sh_level + self.atr_stop_buffer * atr_val
                        dist = sl - cv
                        if dist > 0:
                            position    = -1
                            stop_loss   = sl
                            take_profit = cv - self.rr_target * dist

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


# ── Version 2 — Breaker pullback entry ───────────────────────────────────────

class ICTAMDBreakerStrategy(ICTAMDBase):
    """Wait for pullback into breaker zone after displacement on any lookback."""

    def __init__(self, *args, zone_atr: float = 2.0, max_wait: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.zone_atr = zone_atr
        self.max_wait = max_wait

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return (
            f"ICTAMDBreaker(swing=[{lbs}],eq={self.eq_persist}"
            f",htf={self.htf_sma_period},zone={self.zone_atr}atr"
            f",rr={self.rr_target},wait={self.max_wait})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close        = c["close"]
        high         = c["high"]
        low          = c["low"]
        atr          = c["atr"]
        per_lookback = c["per_lookback"]
        bull_disp    = c["bull_disp"]
        bear_disp    = c["bear_disp"]
        in_session   = c["in_session"]
        warmup       = c["warmup"]

        signals     = pd.Series(0, index=df.index)
        self._stops = pd.Series(float("nan"), index=df.index)
        position    = 0
        stop_loss   = None
        take_profit = None

        pending_long    = False
        pending_short   = False
        pending_zone_lo = None
        pending_zone_hi = None
        pending_stop    = None
        pending_bars    = 0

        for i in range(warmup, len(df)):
            cv      = close.iloc[i]
            lv      = low.iloc[i]
            hv      = high.iloc[i]
            atr_val = atr.iloc[i]

            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss   is not None and cv <= stop_loss:
                    position = 0; stop_loss = None; take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = None; take_profit = None
            elif position == -1:
                if stop_loss   is not None and cv >= stop_loss:
                    position = 0; stop_loss = None; take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = None; take_profit = None

            # Detect setup on any lookback → set pending breaker zone
            if position == 0 and in_session.iloc[i]:
                if bull_disp.iloc[i]:
                    sl_level = _immediate_bull_level(per_lookback, i)
                    if not np.isnan(sl_level) and not pending_long:
                        pending_long    = True
                        pending_short   = False
                        pending_zone_lo = sl_level
                        pending_zone_hi = sl_level + self.zone_atr * atr_val
                        pending_stop    = sl_level - self.atr_stop_buffer * atr_val
                        pending_bars    = 0

                elif not self.long_only and bear_disp.iloc[i]:
                    sh_level = _immediate_bear_level(per_lookback, i)
                    if not np.isnan(sh_level) and not pending_short:
                        pending_short   = True
                        pending_long    = False
                        pending_zone_lo = sh_level - self.zone_atr * atr_val
                        pending_zone_hi = sh_level
                        pending_stop    = sh_level + self.atr_stop_buffer * atr_val
                        pending_bars    = 0

            if pending_long and position == 0:
                pending_bars += 1
                if cv < pending_stop or pending_bars > self.max_wait:
                    pending_long = False
                elif lv <= pending_zone_hi and cv >= pending_zone_lo:
                    dist = cv - pending_stop
                    if dist > 0:
                        position    = 1
                        stop_loss   = pending_stop
                        take_profit = cv + self.rr_target * dist
                    pending_long = False

            elif pending_short and position == 0:
                pending_bars += 1
                if cv > pending_stop or pending_bars > self.max_wait:
                    pending_short = False
                elif hv >= pending_zone_lo and cv <= pending_zone_hi:
                    dist = pending_stop - cv
                    if dist > 0:
                        position    = -1
                        stop_loss   = pending_stop
                        take_profit = cv - self.rr_target * dist
                    pending_short = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")

        return signals


# ── Order block helpers ───────────────────────────────────────────────────────

def _find_ob_long(open_: pd.Series, close_: pd.Series, i: int, lookback: int = 20):
    """Body (lo, hi) of last bearish candle at or before bar i."""
    for j in range(i, max(0, i - lookback), -1):
        if close_.iloc[j] < open_.iloc[j]:
            return float(close_.iloc[j]), float(open_.iloc[j])
    return float("nan"), float("nan")


def _find_ob_short(open_: pd.Series, close_: pd.Series, i: int, lookback: int = 20):
    """Body (lo, hi) of last bullish candle at or before bar i."""
    for j in range(i, max(0, i - lookback), -1):
        if close_.iloc[j] > open_.iloc[j]:
            return float(open_.iloc[j]), float(close_.iloc[j])
    return float("nan"), float("nan")


# ── Entry v1: Break of Structure ──────────────────────────────────────────────

class ICTAMDBOSStrategy(ICTAMDBase):
    """Enter when price closes beyond the displacement candle's high/low (BOS).

    Long:  sweep + bullish displacement → close above displacement high
    Short: sweep + bearish displacement → close below displacement low
    """

    def __init__(self, *args, max_wait: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_wait = max_wait

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return f"ICTAMD_BOS(swing=[{lbs}],rr={self.rr_target},wait={self.max_wait})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close, high, low = c["close"], c["high"], c["low"]
        atr, per_lookback = c["atr"], c["per_lookback"]
        bull_disp, bear_disp = c["bull_disp"], c["bear_disp"]
        in_session, warmup = c["in_session"], c["warmup"]

        signals       = pd.Series(0, index=df.index)
        self._stops   = pd.Series(float("nan"), index=df.index)
        position      = 0
        stop_loss     = take_profit = None
        pending_long  = pending_short = False
        pending_bos_hi = pending_bos_lo = pending_stop = None
        pending_bars  = 0

        for i in range(warmup, len(df)):
            cv, atr_val = close.iloc[i], atr.iloc[i]
            if np.isnan(atr_val):
                signals.iloc[i] = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss is not None and cv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and cv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and in_session.iloc[i]:
                if bull_disp.iloc[i]:
                    sl = _immediate_bull_level(per_lookback, i)
                    if not np.isnan(sl) and not pending_long:
                        pending_long = True; pending_short = False
                        pending_bos_hi = high.iloc[i]
                        pending_stop   = sl - self.atr_stop_buffer * atr_val
                        pending_bars   = 0
                elif not self.long_only and bear_disp.iloc[i]:
                    sh = _immediate_bear_level(per_lookback, i)
                    if not np.isnan(sh) and not pending_short:
                        pending_short = True; pending_long = False
                        pending_bos_lo = low.iloc[i]
                        pending_stop   = sh + self.atr_stop_buffer * atr_val
                        pending_bars   = 0

            if pending_long and position == 0:
                pending_bars += 1
                if cv < pending_stop or pending_bars > self.max_wait:
                    pending_long = False
                elif cv > pending_bos_hi:
                    dist = cv - pending_stop
                    if dist > 0:
                        position = 1; stop_loss = pending_stop
                        take_profit = cv + self.rr_target * dist
                    pending_long = False
            elif pending_short and position == 0:
                pending_bars += 1
                if cv > pending_stop or pending_bars > self.max_wait:
                    pending_short = False
                elif cv < pending_bos_lo:
                    dist = pending_stop - cv
                    if dist > 0:
                        position = -1; stop_loss = pending_stop
                        take_profit = cv - self.rr_target * dist
                    pending_short = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
        return signals


# ── Entry v2: 50% Retracement of Displacement ────────────────────────────────

class ICTAMDMidpointStrategy(ICTAMDBase):
    """Enter when price wicks back to the midpoint of the displacement candle.

    Buying at equilibrium inside the displacement — the tightest ICT entry.
    Stop stays below the swept level.
    """

    def __init__(self, *args, max_wait: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_wait = max_wait

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return f"ICTAMD_50pct(swing=[{lbs}],rr={self.rr_target},wait={self.max_wait})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close, high, low = c["close"], c["high"], c["low"]
        atr, per_lookback = c["atr"], c["per_lookback"]
        bull_disp, bear_disp = c["bull_disp"], c["bear_disp"]
        in_session, warmup = c["in_session"], c["warmup"]

        signals       = pd.Series(0, index=df.index)
        self._stops   = pd.Series(float("nan"), index=df.index)
        position      = 0
        stop_loss     = take_profit = None
        pending_long  = pending_short = False
        pending_mid   = pending_stop = None
        pending_bars  = 0

        for i in range(warmup, len(df)):
            cv, lv, hv = close.iloc[i], low.iloc[i], high.iloc[i]
            atr_val = atr.iloc[i]
            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss is not None and cv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and cv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and in_session.iloc[i]:
                if bull_disp.iloc[i]:
                    sl = _immediate_bull_level(per_lookback, i)
                    if not np.isnan(sl) and not pending_long:
                        pending_long  = True; pending_short = False
                        pending_mid   = (high.iloc[i] + low.iloc[i]) / 2
                        pending_stop  = sl - self.atr_stop_buffer * atr_val
                        pending_bars  = 0
                elif not self.long_only and bear_disp.iloc[i]:
                    sh = _immediate_bear_level(per_lookback, i)
                    if not np.isnan(sh) and not pending_short:
                        pending_short = True; pending_long = False
                        pending_mid   = (high.iloc[i] + low.iloc[i]) / 2
                        pending_stop  = sh + self.atr_stop_buffer * atr_val
                        pending_bars  = 0

            if pending_long and position == 0:
                pending_bars += 1
                if cv < pending_stop or pending_bars > self.max_wait:
                    pending_long = False
                elif lv <= pending_mid:
                    dist = cv - pending_stop
                    if dist > 0:
                        position = 1; stop_loss = pending_stop
                        take_profit = cv + self.rr_target * dist
                    pending_long = False
            elif pending_short and position == 0:
                pending_bars += 1
                if cv > pending_stop or pending_bars > self.max_wait:
                    pending_short = False
                elif hv >= pending_mid:
                    dist = pending_stop - cv
                    if dist > 0:
                        position = -1; stop_loss = pending_stop
                        take_profit = cv - self.rr_target * dist
                    pending_short = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
        return signals


# ── Entry v3: Order Block Re-test ─────────────────────────────────────────────

class ICTAMDOrderBlockStrategy(ICTAMDBase):
    """Enter when price wicks into the body of the order block candle.

    The order block is the last opposing candle before the displacement —
    where institutions placed the orders that caused the move.  Re-testing
    that body is the highest-confluence ICT entry.
    """

    def __init__(self, *args, max_wait: int = 10, ob_lookback: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_wait    = max_wait
        self.ob_lookback = ob_lookback

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return f"ICTAMD_OB(swing=[{lbs}],rr={self.rr_target},wait={self.max_wait})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close, high, low, open_ = c["close"], c["high"], c["low"], c["open_"]
        atr, per_lookback = c["atr"], c["per_lookback"]
        bull_disp, bear_disp = c["bull_disp"], c["bear_disp"]
        in_session, warmup = c["in_session"], c["warmup"]

        signals       = pd.Series(0, index=df.index)
        self._stops   = pd.Series(float("nan"), index=df.index)
        position      = 0
        stop_loss     = take_profit = None
        pending_long  = pending_short = False
        pending_ob_lo = pending_ob_hi = pending_stop = None
        pending_bars  = 0

        for i in range(warmup, len(df)):
            cv, lv, hv = close.iloc[i], low.iloc[i], high.iloc[i]
            atr_val = atr.iloc[i]
            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss is not None and cv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and cv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and in_session.iloc[i]:
                if bull_disp.iloc[i]:
                    sl = _immediate_bull_level(per_lookback, i)
                    if not np.isnan(sl) and not pending_long:
                        ob_lo, ob_hi = _find_ob_long(open_, close, i - 1, self.ob_lookback)
                        if not np.isnan(ob_lo):
                            pending_long  = True; pending_short = False
                            pending_ob_lo = ob_lo; pending_ob_hi = ob_hi
                            pending_stop  = sl - self.atr_stop_buffer * atr_val
                            pending_bars  = 0
                elif not self.long_only and bear_disp.iloc[i]:
                    sh = _immediate_bear_level(per_lookback, i)
                    if not np.isnan(sh) and not pending_short:
                        ob_lo, ob_hi = _find_ob_short(open_, close, i - 1, self.ob_lookback)
                        if not np.isnan(ob_lo):
                            pending_short = True; pending_long = False
                            pending_ob_lo = ob_lo; pending_ob_hi = ob_hi
                            pending_stop  = sh + self.atr_stop_buffer * atr_val
                            pending_bars  = 0

            if pending_long and position == 0:
                pending_bars += 1
                if cv < pending_stop or pending_bars > self.max_wait:
                    pending_long = False
                elif lv <= pending_ob_hi and cv >= pending_ob_lo:
                    dist = cv - pending_stop
                    if dist > 0:
                        position = 1; stop_loss = pending_stop
                        take_profit = cv + self.rr_target * dist
                    pending_long = False
            elif pending_short and position == 0:
                pending_bars += 1
                if cv > pending_stop or pending_bars > self.max_wait:
                    pending_short = False
                elif hv >= pending_ob_lo and cv <= pending_ob_hi:
                    dist = pending_stop - cv
                    if dist > 0:
                        position = -1; stop_loss = pending_stop
                        take_profit = cv - self.rr_target * dist
                    pending_short = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
        return signals


# ── Entry v4: Combo — ANY of BOS / 50% / Order Block ─────────────────────────

class ICTAMDComboStrategy(ICTAMDBase):
    """Takes a trade when ANY of the three entry conditions fires first.

    After AMD detection all three pending entries are armed simultaneously.
    First to trigger wins; the others are cancelled.
    """

    def __init__(self, *args, max_wait: int = 10, ob_lookback: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_wait    = max_wait
        self.ob_lookback = ob_lookback

    @property
    def name(self) -> str:
        lbs = "/".join(str(l) for l in self.swing_lookbacks)
        return f"ICTAMD_Combo(swing=[{lbs}],rr={self.rr_target},wait={self.max_wait})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = self._compute(df)
        close, high, low, open_ = c["close"], c["high"], c["low"], c["open_"]
        atr, per_lookback = c["atr"], c["per_lookback"]
        bull_disp, bear_disp = c["bull_disp"], c["bear_disp"]
        in_session, warmup = c["in_session"], c["warmup"]

        signals       = pd.Series(0, index=df.index)
        self._stops   = pd.Series(float("nan"), index=df.index)
        position      = 0
        stop_loss     = take_profit = None
        pending_long  = pending_short = False
        pending_stop  = None
        pending_bars  = 0
        l_bos_hi = l_mid = l_ob_lo = l_ob_hi = None
        s_bos_lo = s_mid = s_ob_lo = s_ob_hi = None

        for i in range(warmup, len(df)):
            cv, lv, hv = close.iloc[i], low.iloc[i], high.iloc[i]
            atr_val = atr.iloc[i]
            if np.isnan(atr_val):
                signals.iloc[i]     = position
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            if position == 1:
                if stop_loss is not None and cv <= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv >= take_profit:
                    position = 0; stop_loss = take_profit = None
            elif position == -1:
                if stop_loss is not None and cv >= stop_loss:
                    position = 0; stop_loss = take_profit = None
                elif take_profit is not None and cv <= take_profit:
                    position = 0; stop_loss = take_profit = None

            if position == 0 and in_session.iloc[i]:
                if bull_disp.iloc[i]:
                    sl = _immediate_bull_level(per_lookback, i)
                    if not np.isnan(sl) and not pending_long:
                        ob_lo, ob_hi = _find_ob_long(open_, close, i - 1, self.ob_lookback)
                        if not np.isnan(ob_lo):
                            pending_long = True; pending_short = False
                            pending_stop = sl - self.atr_stop_buffer * atr_val
                            pending_bars = 0
                            l_bos_hi = high.iloc[i]
                            l_mid    = (high.iloc[i] + low.iloc[i]) / 2
                            l_ob_lo  = ob_lo; l_ob_hi = ob_hi
                elif not self.long_only and bear_disp.iloc[i]:
                    sh = _immediate_bear_level(per_lookback, i)
                    if not np.isnan(sh) and not pending_short:
                        ob_lo, ob_hi = _find_ob_short(open_, close, i - 1, self.ob_lookback)
                        if not np.isnan(ob_lo):
                            pending_short = True; pending_long = False
                            pending_stop  = sh + self.atr_stop_buffer * atr_val
                            pending_bars  = 0
                            s_bos_lo = low.iloc[i]
                            s_mid    = (high.iloc[i] + low.iloc[i]) / 2
                            s_ob_lo  = ob_lo; s_ob_hi = ob_hi

            if pending_long and position == 0:
                pending_bars += 1
                if cv < pending_stop or pending_bars > self.max_wait:
                    pending_long = False
                else:
                    bos = cv > l_bos_hi
                    mid = lv <= l_mid
                    ob  = l_ob_lo is not None and lv <= l_ob_hi and cv >= l_ob_lo
                    if bos or mid or ob:
                        dist = cv - pending_stop
                        if dist > 0:
                            position = 1; stop_loss = pending_stop
                            take_profit = cv + self.rr_target * dist
                        pending_long = False

            elif pending_short and position == 0:
                pending_bars += 1
                if cv > pending_stop or pending_bars > self.max_wait:
                    pending_short = False
                else:
                    bos = cv < s_bos_lo
                    mid = hv >= s_mid
                    ob  = s_ob_lo is not None and hv >= s_ob_lo and cv <= s_ob_hi
                    if bos or mid or ob:
                        dist = pending_stop - cv
                        if dist > 0:
                            position = -1; stop_loss = pending_stop
                            take_profit = cv - self.rr_target * dist
                        pending_short = False

            signals.iloc[i]     = position
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
        return signals


# ── Indicators ────────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
