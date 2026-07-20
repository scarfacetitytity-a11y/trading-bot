"""AiDEN Index Strategy v2 — bidirectional FVG + OB confluence, all sessions.

Cadre:
  Sage   (MEM-001) — bidirectional gate: H4 EMA bullish = longs only, bearish = shorts only
  Quant  (MEM-002) — dynamic RR: Model 3 + trend strength extend target to 3.5-5.0R
  Builder(MEM-003) — implementation, shorts mirror, regime scoring
  Scout  (MEM-004) — 7-instrument universe confirmed

OS Confluence scoring (max 10 per setup):
  HTF bias confirmed (H4 EMA direction) ......... +2  [hard gate]
  Price in discount (long) / premium (short) .... +1
  Liquidity swept (lows for longs / highs for shorts) +1
  OB + FVG aligned — Model 3 ................... +2
  FVG only — Model 1 ........................... +1
  Session (in active window) ................... +1
  Session prime window (NY first hour 13-15 UTC) +1  [stacks with session]
  RSI pullback zone ............................ +1
  Trend regime (H4 EMA strongly trending) ...... +1
  Asian 50% level (FVG within 20% of pre-session +1  [JP mentor]
    range midpoint — highest-probability London zone)
  Early leakage (pre-session swept prior day     +1  [JP mentor]
    extreme = London reversal sweep setup)
  Inside day + closer target (trade toward       +1  [JP mentor]
    nearer daily high/low when inside day)
  D1 aligned (daily EMA agrees with H4) ......... +1  [optional]

Dynamic RR:
  Base: rr_target (default 2.5)
  Model 3 trade: + rr_model3_bonus (default +0.5)
  Strong H4 trend: + rr_trend_bonus (default +0.5)
  Cap: rr_max (default 5.0)

Parameter units:
  htf_lookback, h4_swing_lookback → H4 BARS (TF-independent after resample)
  Everything else → input-TF bars (scale ×4 for M15 vs H1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


# ── Helpers ───────────────────────────────────────────────────────────────────

# Indicators live in the shared single-source module (strategies/indicators.py).
from strategies.indicators import atr as _atr, rsi as _rsi   # noqa: E402


def _resample_d1(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    d1 = tmp.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    d1 = d1.reset_index().rename(columns={"index": "time"})
    d1["time"] = pd.to_datetime(d1["time"])
    return d1


def _resample_h4(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    h4 = tmp.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    h4 = h4.reset_index().rename(columns={"index": "time"})
    h4["time"] = pd.to_datetime(h4["time"])
    return h4


def _compute_h4_bias_ema(
    h4: pd.DataFrame, fast_n: int, slow_n: int
) -> tuple[pd.Series, pd.Series]:
    """Returns (bias_series, ema_spread_series).

    bias:   1=bullish, -1=bearish, 0=neutral
    spread: (fast-slow)/slow as fraction — magnitude = trend strength
    """
    close  = h4["close"]
    fast   = close.ewm(span=fast_n, adjust=False).mean()
    slow   = close.ewm(span=slow_n, adjust=False).mean()
    spread = (fast - slow) / slow.replace(0, np.nan)

    bias = pd.Series(0, index=h4.index, dtype=int)
    bias[(fast > slow) & (close > fast)] = 1
    bias[(fast < slow) & (close < fast)] = -1
    return bias, spread


def _compute_h4_bias_swing(h4: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    bias  = pd.Series(0, index=h4.index, dtype=int)
    highs = h4["high"].values
    lows  = h4["low"].values
    mid   = lookback // 2

    for i in range(lookback, len(h4)):
        w_hi = highs[i - lookback:i]
        w_lo = lows[i  - lookback:i]
        if w_hi[mid:].max() > w_hi[:mid].max() and w_lo[mid:].min() > w_lo[:mid].min():
            bias.iloc[i] = 1
        elif w_hi[mid:].max() < w_hi[:mid].max() and w_lo[mid:].min() < w_lo[:mid].min():
            bias.iloc[i] = -1

    spread = pd.Series(0.0, index=h4.index)  # swing method has no spread metric
    return bias, spread


def _find_bullish_ob(open_, close, high, low, start_i, lookback):
    """Last bearish candle before start_i (body zone)."""
    for j in range(start_i, max(0, start_i - lookback), -1):
        if close.iloc[j] < open_.iloc[j]:
            return float(min(open_.iloc[j], close.iloc[j])), float(max(open_.iloc[j], close.iloc[j]))
    return None, None


def _find_bearish_ob(open_, close, high, low, start_i, lookback):
    """Last bullish candle before start_i (body zone) — bearish OB for short setups."""
    for j in range(start_i, max(0, start_i - lookback), -1):
        if close.iloc[j] > open_.iloc[j]:
            return float(min(open_.iloc[j], close.iloc[j])), float(max(open_.iloc[j], close.iloc[j]))
    return None, None


def _liq_swept_low(low: pd.Series, i: int, lookback: int) -> bool:
    """Wick below prior N-bar swing low — used for LONG setups."""
    if i < lookback + 2:
        return False
    swing_low = low.iloc[i - lookback:i - 1].min()
    return any(low.iloc[j] < swing_low for j in range(max(0, i - 5), i))


def _liq_swept_high(high: pd.Series, i: int, lookback: int) -> bool:
    """Wick above prior N-bar swing high — used for SHORT setups."""
    if i < lookback + 2:
        return False
    swing_high = high.iloc[i - lookback:i - 1].max()
    return any(high.iloc[j] > swing_high for j in range(max(0, i - 5), i))


# ── Strategy ──────────────────────────────────────────────────────────────────

class AiDENIndexStrategy(Strategy):

    def __init__(
        self,
        # Confluence gate
        min_score: int           = 4,
        # H4 bias — in H4 BARS
        htf_lookback: int        = 20,
        h4_swing_lookback: int   = 40,
        h4_bias_method: str      = "ema",
        discount_pct: float      = 0.5,
        # FVG — in input-TF bars
        min_fvg_atr: float       = 0.10,
        max_fvg_wait: int        = 40,
        max_entry_wait: int      = 8,
        max_active_fvgs: int     = 3,
        # OB
        ob_lookback: int         = 20,
        # Liquidity sweep
        liq_lookback: int        = 10,
        # Risk / RR
        rr_target: float         = 2.5,
        rr_model3_bonus: float   = 0.5,   # extra R when OB+FVG aligned
        rr_trend_bonus: float    = 0.5,   # extra R in strong trend regime
        rr_trend_threshold: float= 0.003, # H4 EMA spread > 0.3% = strong trend
        rr_max: float            = 5.0,
        atr_period: int          = 14,
        atr_stop_buffer: float   = 0.3,
        # RSI
        rsi_period: int          = 14,
        rsi_long_lo: float       = 25.0,  # RSI zone for LONG entries
        rsi_long_hi: float       = 55.0,
        rsi_short_lo: float      = 45.0,  # RSI zone for SHORT entries
        rsi_short_hi: float      = 75.0,
        use_rsi: bool            = True,
        # Session
        session_start: int       = 12,
        session_end: int         = 21,
        session_prime_start: int = 13,    # NY open first hour bonus window
        session_prime_end: int   = 15,
        use_prime_bonus: bool    = True,
        # Volume spike — +1 score when FVG-forming bar volume > mult * 20-bar avg
        use_vol_spike: bool      = False,
        vol_spike_mult: float    = 1.5,
        # CE (Consequent Encroachment) — require price to reach FVG midpoint before entry
        require_ce: bool         = False,
        # D1 bias alignment — +1 score when Daily EMA agrees with H4 EMA
        use_d1_bias: bool        = False,
        # Direction
        long_only: bool          = False,  # False = both longs and shorts
        # Trailing stop — move SL to breakeven once trade reaches +trail_be_r profit
        # Default False: run to full SL/TP (preserves backtest win rate and RR)
        trail_to_be: bool        = False,
        trail_be_r: float        = 1.0,   # R-multiple at which SL moves to entry (BE)
        trail_lock_r: float      = 2.0,   # R-multiple at which SL trails to +1R locked
        # T1 partial close — close t1_partial_pct of position at t1_r, then move SL to BE
        # Default disabled (t1_r=0.0). trail_to_be lock still fires at trail_lock_r.
        t1_r: float              = 0.0,   # R at which T1 fires; 0 = disabled
        t1_partial_pct: float    = 0.5,   # fraction to exit at T1 (0.5 = 50%)
        # Time stop — exit if no TP progress after N bars; 0 = disabled
        time_stop_bars: int      = 0,
    ):
        self.min_score            = min_score
        self.htf_lookback         = htf_lookback
        self.h4_swing_lookback    = h4_swing_lookback
        self.h4_bias_method       = h4_bias_method
        self.discount_pct         = discount_pct
        self.min_fvg_atr          = min_fvg_atr
        self.max_fvg_wait         = max_fvg_wait
        self.max_entry_wait       = max_entry_wait
        self.max_active_fvgs      = max_active_fvgs
        self.ob_lookback          = ob_lookback
        self.liq_lookback         = liq_lookback
        self.rr_target            = rr_target
        self.rr_model3_bonus      = rr_model3_bonus
        self.rr_trend_bonus       = rr_trend_bonus
        self.rr_trend_threshold   = rr_trend_threshold
        self.rr_max               = rr_max
        self.atr_period           = atr_period
        self.atr_stop_buffer      = atr_stop_buffer
        self.rsi_period           = rsi_period
        self.rsi_long_lo          = rsi_long_lo
        self.rsi_long_hi          = rsi_long_hi
        self.rsi_short_lo         = rsi_short_lo
        self.rsi_short_hi         = rsi_short_hi
        self.use_rsi              = use_rsi
        self.session_start        = session_start
        self.session_end          = session_end
        self.session_prime_start  = session_prime_start
        self.session_prime_end    = session_prime_end
        self.use_prime_bonus      = use_prime_bonus
        self.use_vol_spike        = use_vol_spike
        self.vol_spike_mult       = vol_spike_mult
        self.require_ce           = require_ce
        self.use_d1_bias          = use_d1_bias
        self.long_only            = long_only
        self.trail_to_be          = trail_to_be
        self.trail_be_r           = trail_be_r
        self.trail_lock_r         = trail_lock_r
        self.t1_r                 = t1_r
        self.t1_partial_pct       = t1_partial_pct
        self.time_stop_bars       = time_stop_bars
        self._last_h4_bias:       int = 0   # updated on each generate_signals call

    @property
    def name(self) -> str:
        direction = "long" if self.long_only else "bi"
        return (
            f"AiDEN-v2({direction}"
            f",s>={self.min_score}"
            f",{self.h4_bias_method}"
            f",rr={self.rr_target}+dyn"
            f",sess={self.session_start}-{self.session_end})"
        )

    def current_h4_bias(self, df: pd.DataFrame) -> int:
        """Return the current H4 bias (+1/-1/0) from the latest bars.
        Lightweight — just resamples and reads last value. No full loop."""
        h4 = _resample_h4(df)
        if len(h4) < 5:
            return 0
        fast_n = max(self.htf_lookback // 2, 5)
        if self.h4_bias_method == "ema":
            h4_bias, _ = _compute_h4_bias_ema(h4, fast_n, self.htf_lookback)
        else:
            h4_bias, _ = _compute_h4_bias_swing(h4, self.h4_swing_lookback)
        return int(h4_bias.iloc[-1])

    def _dynamic_rr(self, is_model3: bool, trend_strength: float) -> float:
        rr = self.rr_target
        if is_model3:
            rr += self.rr_model3_bonus
        if abs(trend_strength) > self.rr_trend_threshold:
            rr += self.rr_trend_bonus
        return min(rr, self.rr_max)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        open_  = df["open"].astype(float)
        times  = pd.to_datetime(df["time"])
        hours  = times.dt.hour

        atr_s = _atr(high, low, close, self.atr_period)
        self._atr_cache = atr_s
        rsi_s = _rsi(close, self.rsi_period) if self.use_rsi else None

        vol_s    = df["tick_volume"].astype(float) if "tick_volume" in df.columns else None
        vol_mean = vol_s.rolling(20).mean() if vol_s is not None else None

        h4 = _resample_h4(df)
        fast_n = max(self.htf_lookback // 2, 5)

        if self.h4_bias_method == "ema":
            h4_bias, h4_spread = _compute_h4_bias_ema(h4, fast_n, self.htf_lookback)
        else:
            h4_bias, h4_spread = _compute_h4_bias_swing(h4, self.htf_lookback)

        # D1 bias — three-TF alignment gate
        d1_bias_at = None
        if self.use_d1_bias:
            d1 = _resample_d1(df)
            if len(d1) >= 10:
                d1_fast_n = max(3, self.htf_lookback // 4)
                d1_slow_n = max(5, self.htf_lookback // 2)
                d1_b, _   = _compute_h4_bias_ema(d1, d1_fast_n, d1_slow_n)
                d1_times  = d1["time"]
                def _d1_bias_at(bar_time, _d1b=d1_b, _d1t=d1_times):
                    idx = _d1t.searchsorted(bar_time, side="right") - 1
                    return int(_d1b.iloc[idx]) if idx >= 0 else 0
                d1_bias_at = _d1_bias_at

        # ── Pre-session range (JP mentor: Asian 50% level, early leakage, inside day) ──
        # For each UTC day, compute the high/low of bars BEFORE session_start (the
        # "Asian" or pre-market range) and the previous day's high/low. Three new
        # confluences are derived from these at FVG detection time (below).
        _dates_arr       = times.dt.normalize()
        _all_dates_list  = sorted(_dates_arr.unique())
        _ph  = np.full(len(df), np.nan)
        _pl  = np.full(len(df), np.nan)
        _pdh = np.full(len(df), np.nan)  # prior day high
        _pdl = np.full(len(df), np.nan)  # prior day low
        _date_idx_map: dict = {}
        for _d in _all_dates_list:
            _date_idx_map[_d] = np.where((_dates_arr == _d).values)[0]
        for _k, _d in enumerate(_all_dates_list):
            _idxs = _date_idx_map[_d]
            _pre  = np.where(hours.iloc[_idxs].values < self.session_start)[0]
            if len(_pre):
                _ph[_idxs] = float(high.iloc[_idxs[_pre]].max())
                _pl[_idxs] = float(low.iloc[_idxs[_pre]].min())
            if _k > 0:
                _pi = _date_idx_map[_all_dates_list[_k - 1]]
                _pdh[_idxs] = float(high.iloc[_pi].max())
                _pdl[_idxs] = float(low.iloc[_pi].min())
        _presess_mid = (_ph + _pl) / 2.0
        _presess_rng = _ph - _pl

        h4["bias"]   = h4_bias.values
        h4["spread"] = h4_spread.values
        h4_times     = h4["time"]

        def _h4_at(bar_time):
            idx = h4_times.searchsorted(bar_time, side="right") - 1
            if idx < 0:
                return 0, 0.0
            return int(h4["bias"].iloc[idx]), float(h4["spread"].iloc[idx])

        def _swing_range_at(bar_time):
            idx = h4_times.searchsorted(bar_time, side="right") - 1
            if idx < self.h4_swing_lookback:
                return float("nan"), float("nan")
            w = h4.iloc[idx - self.h4_swing_lookback:idx]
            return float(w["high"].max()), float(w["low"].min())

        signals      = pd.Series(0.0, index=df.index)
        self._stops  = pd.Series(float("nan"), index=df.index)
        self._scores = pd.Series(0, index=df.index)
        self._score_reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

        position         = 0
        position_size    = 0.0
        stop_loss        = None
        initial_sl       = None   # SL at entry — never modified; used for R calculations
        take_profit      = None
        entry_price      = None
        t1_hit           = False
        bars_since_entry = 0
        active_fvgs: list[dict] = []

        warmup = max(self.atr_period + 3, self.ob_lookback, self.liq_lookback,
                     self.rsi_period + 2 if self.use_rsi else 0)

        for i in range(warmup, len(df)):
            cv       = float(close.iloc[i])
            lv       = float(low.iloc[i])
            hv       = float(high.iloc[i])
            atr_val  = float(atr_s.iloc[i])
            bar_time = times.iloc[i]
            hour     = int(hours.iloc[i])

            if np.isnan(atr_val):
                signals.iloc[i]     = position * position_size
                self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
                continue

            # ── 1. Manage open position ───────────────────────────────────
            if position == 1:
                bars_since_entry += 1

                # Trail — reference initial_sl so trail survives T1 BE move
                if self.trail_to_be and initial_sl is not None and entry_price is not None:
                    init_risk = entry_price - initial_sl
                    if init_risk > 0:
                        if cv >= entry_price + self.trail_lock_r * init_risk:
                            stop_loss = max(stop_loss, entry_price + init_risk)
                        elif cv >= entry_price + self.trail_be_r * init_risk and not t1_hit:
                            stop_loss = max(stop_loss, entry_price)

                # T1 partial — fires once at t1_r; moves SL to BE for runner
                if self.t1_r > 0 and not t1_hit and initial_sl is not None:
                    init_risk = entry_price - initial_sl
                    if init_risk > 0 and cv >= entry_price + self.t1_r * init_risk:
                        t1_hit = True
                        position_size = 1.0 - self.t1_partial_pct
                        stop_loss = entry_price

                # Time stop — exit if no progress after N bars
                if self.time_stop_bars > 0 and bars_since_entry >= self.time_stop_bars:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif stop_loss is not None and lv <= stop_loss:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif take_profit is not None and hv >= take_profit:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0

            elif position == -1:
                bars_since_entry += 1

                if self.trail_to_be and initial_sl is not None and entry_price is not None:
                    init_risk = initial_sl - entry_price
                    if init_risk > 0:
                        if cv <= entry_price - self.trail_lock_r * init_risk:
                            stop_loss = min(stop_loss, entry_price - init_risk)
                        elif cv <= entry_price - self.trail_be_r * init_risk and not t1_hit:
                            stop_loss = min(stop_loss, entry_price)

                if self.t1_r > 0 and not t1_hit and initial_sl is not None:
                    init_risk = initial_sl - entry_price
                    if init_risk > 0 and cv <= entry_price - self.t1_r * init_risk:
                        t1_hit = True
                        position_size = 1.0 - self.t1_partial_pct
                        stop_loss = entry_price

                if self.time_stop_bars > 0 and bars_since_entry >= self.time_stop_bars:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif stop_loss is not None and hv >= stop_loss:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0
                elif take_profit is not None and lv <= take_profit:
                    position = 0; position_size = 0.0; t1_hit = False
                    stop_loss = take_profit = entry_price = initial_sl = None
                    bars_since_entry = 0

            if position == 0 and i >= warmup + 2:
                htf_bias, trend_strength = _h4_at(bar_time)
                swing_hi, swing_lo       = _swing_range_at(bar_time)
                in_session   = self.session_start <= hour < self.session_end
                in_prime     = self.session_prime_start <= hour < self.session_prime_end
                rsi_val      = float(rsi_s.iloc[i]) if rsi_s is not None else float("nan")
                strongly_trending = abs(trend_strength) > self.rr_trend_threshold

                # Remove invalidated FVGs
                if htf_bias == 0:
                    self._expire_fvgs_neutral(active_fvgs, cv)
                    signals.iloc[i]     = position * position_size
                    self._stops.iloc[i] = float("nan")
                    continue

                # ── 2. Detect new FVGs ───────────────────────────────────
                h2  = float(high.iloc[i - 2])
                l2  = float(low.iloc[i - 2])

                # LONG setup — bullish FVG
                if htf_bias == 1:
                    bull_gap = lv - h2
                    if bull_gap >= self.min_fvg_atr * atr_val:
                        score = 2; reasons = ["H4 bias +2"]  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv < mid:
                                score += 1; reasons.append("Discount zone")

                        if _liq_swept_low(low, i, self.liq_lookback):
                            score += 1; reasons.append("Liquidity sweep")

                        ob_lo, ob_hi = _find_bullish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, lv) - max(ob_lo, h2) > 0:
                            score += 2; is_model3 = True; reasons.append("Order block (M3) +2")
                        else:
                            score += 1; is_model3 = False; reasons.append("Order block")

                        if in_session:
                            score += 1; reasons.append("Session")
                        if in_prime and self.use_prime_bonus:
                            score += 1; reasons.append("Prime window")

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_long_lo <= rsi_val <= self.rsi_long_hi:
                                score += 1; reasons.append("RSI zone")

                        if strongly_trending:
                            score += 1; reasons.append("Trend regime")

                        if self.use_vol_spike and vol_mean is not None:
                            vm = float(vol_mean.iloc[i])
                            if vm > 0 and float(vol_s.iloc[i]) > self.vol_spike_mult * vm:
                                score += 1; reasons.append("Volume spike")

                        if d1_bias_at is not None and d1_bias_at(bar_time) == 1:
                            score += 1; reasons.append("D1 aligned")

                        # ── JP Mentor confluences (Asian session structure) ──
                        _psm = _presess_mid[i]; _psr = _presess_rng[i]
                        _pdh_i = _pdh[i];       _pdl_i = _pdl[i]
                        # Asian 50% level: FVG formed within 20% of pre-session range
                        # around the midpoint — highest-probability London entry zone
                        if not np.isnan(_psm) and _psr > 0:
                            if abs(cv - _psm) <= 0.20 * _psr:
                                score += 1; reasons.append("Asian 50% level")
                            # Early leakage (long): pre-session swept prior day lows before
                            # London opened — London completing the sweep = reversal buy
                            if not np.isnan(_pdl_i) and _pl[i] < _pdl_i:
                                score += 1; reasons.append("Early leakage (London sweep)")
                        # Inside day + closer target: trade toward the nearer daily extreme
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and hv < _pdh_i and lv > _pdl_i
                                and (_pdh_i - cv) < (cv - _pdl_i)):
                            score += 1; reasons.append("Inside day (closer high)")

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bull",
                                "fvg_lo":   h2,
                                "fvg_hi":   lv,
                                "fvg_ce":   (h2 + lv) / 2,
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "reasons":  reasons,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })


                # SHORT setup — bearish FVG
                if htf_bias == -1 and not self.long_only:
                    bear_gap = l2 - hv
                    if bear_gap >= self.min_fvg_atr * atr_val:
                        score = 2; reasons = ["H4 bias +2"]  # HTF bias +2

                        if not np.isnan(swing_hi) and swing_hi > swing_lo:
                            mid = swing_lo + (swing_hi - swing_lo) * self.discount_pct
                            if cv > mid:
                                score += 1; reasons.append("Premium zone")

                        if _liq_swept_high(high, i, self.liq_lookback):
                            score += 1; reasons.append("Liquidity sweep")

                        ob_lo, ob_hi = _find_bearish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        if ob_lo is not None and min(ob_hi, l2) - max(ob_lo, hv) > 0:
                            score += 2; is_model3 = True; reasons.append("Order block (M3) +2")
                        else:
                            score += 1; is_model3 = False; reasons.append("Order block")

                        if in_session:
                            score += 1; reasons.append("Session")
                        if in_prime and self.use_prime_bonus:
                            score += 1; reasons.append("Prime window")

                        if self.use_rsi and not np.isnan(rsi_val):
                            if self.rsi_short_lo <= rsi_val <= self.rsi_short_hi:
                                score += 1; reasons.append("RSI zone")

                        if strongly_trending:
                            score += 1; reasons.append("Trend regime")

                        if self.use_vol_spike and vol_mean is not None:
                            vm = float(vol_mean.iloc[i])
                            if vm > 0 and float(vol_s.iloc[i]) > self.vol_spike_mult * vm:
                                score += 1; reasons.append("Volume spike")

                        if d1_bias_at is not None and d1_bias_at(bar_time) == -1:
                            score += 1; reasons.append("D1 aligned")

                        # ── JP Mentor confluences (Asian session structure) ──
                        _psm = _presess_mid[i]; _psr = _presess_rng[i]
                        _pdh_i = _pdh[i];       _pdl_i = _pdl[i]
                        if not np.isnan(_psm) and _psr > 0:
                            if abs(cv - _psm) <= 0.20 * _psr:
                                score += 1; reasons.append("Asian 50% level")
                            # Early leakage (short): pre-session swept prior day highs —
                            # London completing the sweep = reversal sell
                            if not np.isnan(_pdh_i) and _ph[i] > _pdh_i:
                                score += 1; reasons.append("Early leakage (London sweep)")
                        if (not np.isnan(_pdh_i) and not np.isnan(_pdl_i)
                                and hv < _pdh_i and lv > _pdl_i
                                and (cv - _pdl_i) < (_pdh_i - cv)):
                            score += 1; reasons.append("Inside day (closer low)")

                        if score >= self.min_score:
                            active_fvgs.append({
                                "dir":      "bear",
                                "fvg_lo":   hv,
                                "fvg_hi":   l2,
                                "fvg_ce":   (hv + l2) / 2,
                                "ob_lo":    ob_lo,
                                "ob_hi":    ob_hi,
                                "score":    score,
                                "reasons":  reasons,
                                "formed":   i,
                                "tested":   False,
                                "test_bar": None,
                                "model3":   is_model3,
                                "trend_s":  trend_strength,
                            })

                # Trim FVG queue
                if len(active_fvgs) > self.max_active_fvgs * 2:
                    active_fvgs.sort(key=lambda x: x["score"], reverse=True)
                    active_fvgs = active_fvgs[:self.max_active_fvgs * 2]

                # ── 3. Process queued FVGs ───────────────────────────────
                to_remove = []
                for fvg in active_fvgs:
                    fvg_lo = fvg["fvg_lo"]
                    fvg_hi = fvg["fvg_hi"]

                    if (i - fvg["formed"]) > self.max_fvg_wait:
                        to_remove.append(fvg); continue

                    if fvg["dir"] == "bull":
                        if cv < fvg_lo:
                            to_remove.append(fvg); continue
                        ce_level = fvg.get("fvg_ce", fvg_hi) if self.require_ce else fvg_hi
                        if not fvg["tested"] and lv <= ce_level:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv > fvg_hi and position == 0:
                                stop_anchor = fvg["ob_lo"] if fvg["ob_lo"] is not None else fvg_lo
                                sl   = stop_anchor - self.atr_stop_buffer * atr_val
                                dist = cv - sl
                                if dist > 0:
                                    rr               = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position         = 1
                                    position_size    = 1.0
                                    stop_loss        = sl
                                    initial_sl       = sl
                                    take_profit      = cv + rr * dist
                                    entry_price      = cv
                                    t1_hit           = False
                                    bars_since_entry = 0
                                    self._scores.iloc[i] = fvg["score"]
                                    self._score_reasons.iloc[i] = fvg.get("reasons", [])
                                to_remove.append(fvg)

                    else:  # bear
                        if cv > fvg_hi:
                            to_remove.append(fvg); continue
                        ce_level = fvg.get("fvg_ce", fvg_lo) if self.require_ce else fvg_lo
                        if not fvg["tested"] and hv >= ce_level:
                            fvg["tested"] = True; fvg["test_bar"] = i
                        if fvg["tested"]:
                            if (i - fvg["test_bar"]) > self.max_entry_wait:
                                to_remove.append(fvg)
                            elif cv < fvg_lo and position == 0:
                                stop_anchor = fvg["ob_hi"] if fvg["ob_hi"] is not None else fvg_hi
                                sl   = stop_anchor + self.atr_stop_buffer * atr_val
                                dist = sl - cv
                                if dist > 0:
                                    rr               = self._dynamic_rr(fvg["model3"], fvg["trend_s"])
                                    position         = -1
                                    position_size    = 1.0
                                    stop_loss        = sl
                                    initial_sl       = sl
                                    take_profit      = cv - rr * dist
                                    entry_price      = cv
                                    t1_hit           = False
                                    bars_since_entry = 0
                                    self._scores.iloc[i] = fvg["score"]
                                    self._score_reasons.iloc[i] = fvg.get("reasons", [])
                                to_remove.append(fvg)

                for fvg in to_remove:
                    if fvg in active_fvgs:
                        active_fvgs.remove(fvg)

            signals.iloc[i]     = position * position_size
            self._stops.iloc[i] = stop_loss if stop_loss is not None else float("nan")
            if position != 0 and i > 0 and self._scores.iloc[i] == 0:
                self._scores.iloc[i] = self._scores.iloc[i - 1]
                self._score_reasons.iloc[i] = self._score_reasons.iloc[i - 1]

        return signals

    def _expire_fvgs_neutral(self, active_fvgs: list, cv: float) -> None:
        to_remove = [
            f for f in active_fvgs
            if (f["dir"] == "bull" and cv < f["fvg_lo"]) or
               (f["dir"] == "bear" and cv > f["fvg_hi"])
        ]
        for f in to_remove:
            active_fvgs.remove(f)
