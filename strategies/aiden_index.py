"""AiDEN Index Strategy v2 — bidirectional FVG + OB confluence, all sessions.

Cadre:
  Sage   (MEM-001) — bidirectional gate: H4 EMA bullish = longs only, bearish = shorts only
  Quant  (MEM-002) — dynamic RR: Model 3 + trend strength extend target to 3.5-5.0R
  Builder(MEM-003) — implementation, shorts mirror, regime scoring
  Scout  (MEM-004) — 7-instrument universe confirmed

OS Confluence scoring:
  HTF bias confirmed (H4 EMA direction) ......... +2  [hard gate]
  Price in discount (long) / premium (short) .... +1
  H4 impulse 50% (within 10% of H4 swing mid) .. +1  [JP mentor v5 — "50% is always a POI"]
  Liquidity swept (lows for longs / highs for shorts) +1
  OB + FVG aligned — Model 3 ................... +2
  FVG only — Model 1 ........................... +1
  Session (in active window) ................... +1
  Session prime window (NY first hour 13-15 UTC) +1  [stacks with session]
  RSI pullback zone ............................ +1
  Trend regime (H4 EMA strongly trending) ...... +1
  Manipulation W (long) / M (short) ............ +1  [JP mentor — ICC entry model]
  Asian 50% level (within 20% of Asian H/L mid)  +1  [JP mentor]
  London 50% NY POI (NY session + at London mid) +1  [JP mentor v4]
  Early leakage (Asian range broken before London)+1  [JP mentor]
  Inside day + closer target ................... +1  [JP mentor]
  PDH/PDL swept today ......................... +1  [JP mentor v2]
  D1 FVG retest (price inside daily imbalance)   +1  [JP mentor v2]
  D1 aligned (daily EMA agrees with H4) ......... +1  [optional]
  Volume spike ................................. +1  [optional]
  W1 FVG retest (weekly imbalance zone) ......... +2  [JP mentor v3]
  Weekly discount/premium (range position) ...... +1  [JP mentor v7]
  NY swept London Low/High ..................... +1  [JP mentor v7]
  Double bottom/top cluster at FVG ............. +1  [JP mentor v9]
  Multi-TF synchrony (H4 or D1 close alignment) +1  [JP mentor — candle close confluence]
  USDJPY macro aligned (DXY-linked instruments)  +1  [JP mentor v7 — UJ as DXY proxy]
  Previous battlefield (prior congestion CHoCH)   +1  [JP mentor v10]

Negative confluences (subtract from score):
  No liquidity sweep                             -1  (market uncleared)
  Opposing manipulation pattern                  -1  (M against long / W against short)
  RSI extreme against trade                      -1  (>75 for long / <25 for short)
  Untaken session liquidity in trade path        -1  (session low not swept for long /
                                                      session high not swept for short —
                                                      magnet pulls price there first)
  Both Asian H+L swept early                     -1  (JP mentor v7: "leaked early, always
                                                      a concern" — directional clarity lost)
  Mid daily range (40-60% of prior day H/L)      -1  (JP mentor v7: "market could go up or
                                                      down, I'm a bit concerned about that"
                                                      — no premium/discount edge available)

Order Block detection (upgraded v7):
  Requires wick on the OB candle + next candle body fully engulfs the OB body.
  JP mentor v7: "There has to be some wick sticking out here. It needs to touch a
  red wick and then the next candle swallows that candle — that becomes an order block."

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
from strategies.indicators import atr as _atr, rsi as _rsi, bollinger_bands as _bb   # noqa: E402

# Per-instrument active session windows (UTC hours, inclusive start exclusive end).
# Session confluence fires when the bar hour falls inside ANY of the listed ranges.
# An instrument with multiple active windows (e.g. JPY trades London + Asia) can
# list both. Gold/Silver are 24h so any hour qualifies.
_INSTRUMENT_SESSIONS: dict[str, list[tuple[int, int]]] = {
    # Forex — London + NY overlap
    "GBPUSD":     [(7, 17)],
    "EURUSD":     [(7, 17)],
    "USDJPY":     [(0, 8), (12, 17)],   # Asia + NY overlap
    # Metals — near 24h, weight toward NY
    "XAUUSD":     [(0, 22)],
    "XAGUSD":     [(0, 22)],
    # US indices — NY only
    "US30":       [(13, 21)],
    "US100":      [(13, 21)],
    "US500":      [(13, 21)],
    "US2000":     [(13, 21)],
    # Asian indices — Tokyo session
    "JP225":      [(0, 8)],
    "HK50":       [(1, 9)],
}
_DEFAULT_SESSION: list[tuple[int, int]] = [(7, 21)]   # fallback for unknown symbols


def _active_session(symbol: str, hour: int) -> bool:
    """Return True if hour is within any active window for the symbol."""
    key = symbol.replace(".cash", "").replace(".fx", "").upper()
    for pat, windows in _INSTRUMENT_SESSIONS.items():
        if key.startswith(pat):
            return any(s <= hour < e for s, e in windows)
    return any(s <= hour < e for s, e in _DEFAULT_SESSION)


# Index CFDs have genuine liquidity gaps outside exchange hours — hard session gate
# remains active for these. Forex/metals trade near-24h so score alone decides.
_LIQUIDITY_GATED = {"US30", "US100", "US500", "US2000", "UK100", "JP225", "HK50"}


def _needs_session_gate(symbol: str) -> bool:
    key = symbol.replace(".cash", "").replace(".fx", "").upper()
    return any(key.startswith(s) for s in _LIQUIDITY_GATED)


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


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    tmp   = df[["open", "high", "low", "close"]].copy()
    tmp.index = times
    w1 = tmp.resample("W-MON", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    w1 = w1.reset_index().rename(columns={"index": "time"})
    w1["time"] = pd.to_datetime(w1["time"])
    return w1


# ── Phase 0 extraction — pure detectors moved verbatim to analysis/ ───────────
# Re-imported here so every existing import path (e.g. `from strategies.aiden_index
# import _compute_h4_bias_ema`) and all internal call sites keep working unchanged.
from analysis.structure import (   # noqa: E402,F401
    _compute_h4_bias_ema, _compute_h4_bias_swing,
    _find_bullish_ob, _find_bearish_ob,
    _is_prior_battlefield, _bos_bull, _bos_bear, _alt_dup,
)
from analysis.liquidity import (   # noqa: E402,F401
    _liq_swept_low, _liq_swept_high,
    _sweep_reversal_bull, _sweep_reversal_bear,
    _manipulation_w, _manipulation_m,
)


# ── Strategy ──────────────────────────────────────────────────────────────────

class AiDENIndexStrategy(Strategy):

    def __init__(
        self,
        # Confluence gate
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
        h4_bias_gate: bool       = True,   # False = sniper mode: build FVGs both ways, H4 bias is confluence not gate
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
        # Bollinger Bands — +1 score when BB confirms premium/discount zone at FVG detection
        # +1 additional when BB is in squeeze (bandwidth < bb_squeeze_pct of mid)
        use_bb: bool             = True,
        bb_period: int           = 20,
        bb_std: float            = 2.0,
        bb_squeeze_pct: float    = 0.04,  # bandwidth / mid < 4% = squeeze
    ):
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
        self.h4_bias_gate         = h4_bias_gate
        self.trail_to_be          = trail_to_be
        self.trail_be_r           = trail_be_r
        self.trail_lock_r         = trail_lock_r
        self.t1_r                 = t1_r
        self.t1_partial_pct       = t1_partial_pct
        self.time_stop_bars       = time_stop_bars
        self.use_bb               = use_bb
        self.bb_period            = bb_period
        self.bb_std               = bb_std
        self.bb_squeeze_pct       = bb_squeeze_pct
        self._last_h4_bias:       int = 0   # updated on each generate_signals call
        # Gap 7 — USDJPY macro H4 bias; set externally by orchestrator before each bar.
        # +1 = USD trending up (JPY weak), -1 = USD trending down, 0 = neutral/unknown.
        self._uj_h4_bias:         int = 0
        # Set by orchestrator at startup so session windows are instrument-aware.
        self._symbol:             str = ""

    @property
    def name(self) -> str:
        direction = "long" if self.long_only else "bi"
        return (
            f"AiDEN-v2({direction}"
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
        _t     = pd.to_datetime(df["time"])
        times  = _t.dt.tz_localize(None) if _t.dt.tz is not None else _t
        hours  = times.dt.hour

        atr_s = _atr(high, low, close, self.atr_period)
        self._atr_cache = atr_s
        rsi_s = _rsi(close, self.rsi_period) if self.use_rsi else None
        bb_upper_s, bb_mid_s, bb_lower_s, bb_bw_s = (
            _bb(close, self.bb_period, self.bb_std) if self.use_bb
            else (None, None, None, None)
        )

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
                _d1t_raw  = pd.to_datetime(d1["time"])
                d1_times  = _d1t_raw.dt.tz_localize(None) if _d1t_raw.dt.tz is not None else _d1t_raw
                def _d1_bias_at(bar_time, _d1b=d1_b, _d1t=d1_times):
                    _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
                    idx = _d1t.searchsorted(_bt, side="right") - 1
                    return int(_d1b.iloc[idx]) if idx >= 0 else 0
                d1_bias_at = _d1_bias_at

        # ── D1 FVG zones (mentor: "I marked out the imbalance between candles on the daily") ──
        # Bullish D1 FVG: gap between bar[i-2].high and bar[i].low (bar i's low > bar i-2's high).
        # Bearish D1 FVG: gap between bar[i-2].low and bar[i].high (bar i's high < bar i-2's low).
        # Computed once from D1 resample; last 7 D1 bars scanned.
        _d1_fvg_zones: list[tuple[float, float, int]] = []  # (lo, hi, direction)
        _d1_for_fvg = _resample_d1(df)
        if len(_d1_for_fvg) >= 3:
            _d1h = _d1_for_fvg["high"].astype(float)
            _d1l = _d1_for_fvg["low"].astype(float)
            for _dj in range(2, min(len(_d1_for_fvg), 9)):
                _bull_gap_d1 = _d1l.iloc[_dj] - _d1h.iloc[_dj - 2]
                if _bull_gap_d1 > 0:
                    _d1_fvg_zones.append((_d1h.iloc[_dj - 2], _d1l.iloc[_dj], 1))
                _bear_gap_d1 = _d1l.iloc[_dj - 2] - _d1h.iloc[_dj]
                if _bear_gap_d1 > 0:
                    _d1_fvg_zones.append((_d1h.iloc[_dj], _d1l.iloc[_dj - 2], -1))

        # ── Weekly FVG zones + range (JP mentor: "closing up weekly imbalance") ──
        # Weekly FVGs are major draw targets — the largest-TF unfilled institutional
        # imbalance. +2 score when price is retesting a weekly FVG zone.
        # Weekly mid/range used for weekly premium/discount score (+1 / -1).
        _w1_fvg_zones: list[tuple[float, float, int]] = []  # (lo, hi, direction)
        _w1_mid_arr  = np.full(len(df), np.nan)
        _w1_rng_arr  = np.full(len(df), np.nan)
        _w1_df = _resample_weekly(df)
        if len(_w1_df) >= 3:
            _w1h = _w1_df["high"].astype(float)
            _w1l = _w1_df["low"].astype(float)
            for _wj in range(2, min(len(_w1_df), 6)):  # last 5 weekly candles
                _bull_gap_w1 = _w1l.iloc[_wj] - _w1h.iloc[_wj - 2]
                if _bull_gap_w1 > 0:
                    _w1_fvg_zones.append((_w1h.iloc[_wj - 2], _w1l.iloc[_wj], 1))
                _bear_gap_w1 = _w1l.iloc[_wj - 2] - _w1h.iloc[_wj]
                if _bear_gap_w1 > 0:
                    _w1_fvg_zones.append((_w1h.iloc[_wj], _w1l.iloc[_wj - 2], -1))
        # Current week mid/range for each bar — map weekly rows onto input bars
        _w1t_raw  = pd.to_datetime(_w1_df["time"]) if len(_w1_df) else pd.Series([], dtype="datetime64[ns]")
        _w1_times = _w1t_raw.dt.tz_localize(None) if len(_w1_df) and _w1t_raw.dt.tz is not None else _w1t_raw
        for _wi in range(len(_w1_df)):
            _wstart = _w1_times.iloc[_wi]
            _wend   = _w1_times.iloc[_wi + 1] if _wi + 1 < len(_w1_df) else pd.Timestamp.max
            _wmask  = (times >= _wstart) & (times < _wend)
            _wh     = float(_w1_df["high"].iloc[_wi])
            _wl     = float(_w1_df["low"].iloc[_wi])
            _w1_mid_arr[_wmask.values] = (_wh + _wl) / 2.0
            _w1_rng_arr[_wmask.values] = _wh - _wl

        # ── Pre-session range (JP mentor: Asian 50% level, early leakage, inside day) ──
        # For each UTC day, compute the high/low of bars BEFORE session_start (the
        # "Asian" or pre-market range) and the previous day's high/low. Three new
        # confluences are derived from these at FVG detection time (below).
        # _cdh/_cdl: running daily high/low up to each bar (for PDH/PDL sweep check).
        _dates_arr       = times.dt.normalize()
        _all_dates_list  = sorted(_dates_arr.unique())
        _ph  = np.full(len(df), np.nan)
        _pl  = np.full(len(df), np.nan)
        _pdh = np.full(len(df), np.nan)  # prior day high
        _pdl = np.full(len(df), np.nan)  # prior day low
        _cdh = np.full(len(df), np.nan)  # current day running high at bar i
        _cdl = np.full(len(df), np.nan)  # current day running low at bar i
        _date_idx_map: dict = {}
        for _d in _all_dates_list:
            _date_idx_map[_d] = np.where((_dates_arr == _d).values)[0]
        for _k, _d in enumerate(_all_dates_list):
            _idxs = _date_idx_map[_d]
            _pre  = np.where(hours.iloc[_idxs].values < self.session_start)[0]
            if len(_pre):
                _ph[_idxs] = float(high.iloc[_idxs[_pre]].max())
                _pl[_idxs] = float(low.iloc[_idxs[_pre]].min())
            # Running cumulative high/low within each day
            _hi_vals = high.iloc[_idxs].values
            _lo_vals = low.iloc[_idxs].values
            for _j in range(len(_idxs)):
                _cdh[_idxs[_j]] = float(np.max(_hi_vals[:_j + 1]))
                _cdl[_idxs[_j]] = float(np.min(_lo_vals[:_j + 1]))
            if _k > 0:
                _pi = _date_idx_map[_all_dates_list[_k - 1]]
                _pdh[_idxs] = float(high.iloc[_pi].max())
                _pdl[_idxs] = float(low.iloc[_pi].min())
        _presess_mid = (_ph + _pl) / 2.0
        _presess_rng = _ph - _pl

        # ── London session range (JP mentor video 4: NY taps into 50% of London range) ──
        # London session: 07:00–12:00 UTC. When NY opens (hour >= 13) and price is at
        # 50% of today's London H/L → strong POI. "NY pushed up to 50% of previous session
        # (London) and had huge wick rejection, then continued London's bearish energy."
        _lon_ph = np.full(len(df), np.nan)  # London session high per bar
        _lon_pl = np.full(len(df), np.nan)  # London session low per bar
        _london_start_h, _london_end_h = 7, 12
        for _k, _d in enumerate(_all_dates_list):
            _idxs  = _date_idx_map[_d]
            _lon_m = np.where(
                (hours.iloc[_idxs].values >= _london_start_h) &
                (hours.iloc[_idxs].values < _london_end_h)
            )[0]
            if len(_lon_m):
                _lh = float(high.iloc[_idxs[_lon_m]].max())
                _ll = float(low.iloc[_idxs[_lon_m]].min())
                _lon_ph[_idxs] = _lh
                _lon_pl[_idxs] = _ll
        _lon_mid = (_lon_ph + _lon_pl) / 2.0
        _lon_rng = _lon_ph - _lon_pl

        h4["bias"]   = h4_bias.values
        h4["spread"] = h4_spread.values
        _h4t_raw = pd.to_datetime(h4["time"])
        h4_times = _h4t_raw.dt.tz_localize(None) if _h4t_raw.dt.tz is not None else _h4t_raw

        def _h4_at(bar_time):
            _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
            idx = h4_times.searchsorted(_bt, side="right") - 1
            if idx < 0:
                return 0, 0.0
            return int(h4["bias"].iloc[idx]), float(h4["spread"].iloc[idx])

        def _swing_range_at(bar_time):
            _bt = bar_time.replace(tzinfo=None) if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None else bar_time
            idx = h4_times.searchsorted(_bt, side="right") - 1
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
                     self.rsi_period + 2 if self.use_rsi else 0,
                     self.bb_period if self.use_bb else 0)

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
                in_session   = _active_session(self._symbol, hour)
                in_prime     = self.session_prime_start <= hour < self.session_prime_end
                rsi_val      = float(rsi_s.iloc[i]) if rsi_s is not None else float("nan")
                strongly_trending = abs(trend_strength) > self.rr_trend_threshold

                # Remove invalidated FVGs — expire any FVG whose bias no longer matches.
                # Sniper mode (h4_bias_gate=False): keep FVGs regardless of H4 flip —
                # price structure is the authority, not the trend filter.
                if self.h4_bias_gate:
                    to_expire = []
                    for _fvg in active_fvgs:
                        if _fvg["dir"] == "bull" and htf_bias != 1:
                            to_expire.append(_fvg)
                        elif _fvg["dir"] == "bear" and htf_bias != -1:
                            to_expire.append(_fvg)
                    for _fvg in to_expire:
                        active_fvgs.remove(_fvg)

                if htf_bias == 0 and self.h4_bias_gate:
                    self._expire_fvgs_neutral(active_fvgs, cv)
                    signals.iloc[i]     = position * position_size
                    self._stops.iloc[i] = float("nan")
                    continue

                # ── 2. Detect new FVGs ───────────────────────────────────
                h2  = float(high.iloc[i - 2])
                l2  = float(low.iloc[i - 2])

                # Weekly premium/discount hard gate — block longs above weekly 50%,
                # shorts below it. Swing-range check alone let gold longs through at
                # weekly premium when the weekly range dwarfed the H4 swing.
                _w1_mid_i       = _w1_mid_arr[i]
                _w1_in_premium  = not np.isnan(_w1_mid_i) and cv > _w1_mid_i
                _w1_in_discount = not np.isnan(_w1_mid_i) and cv < _w1_mid_i

                # LONG setup — bullish FVG
                # Sniper mode: build long FVG regardless of H4 bias; bias becomes confluence only
                if htf_bias == 1 or not self.h4_bias_gate:
                    bull_gap = lv - h2
                    if bull_gap >= self.min_fvg_atr * atr_val:
                        ob_lo, ob_hi = _find_bullish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        is_model3 = ob_lo is not None and min(ob_hi, lv) - max(ob_lo, h2) > 0
                        reasons = ["FVG"]
                        score = 0
                        if ob_lo is not None:
                            reasons.append("OB")
                            score += 1
                        if self.use_bb and bb_mid_s is not None:
                            _bb_mid_i = float(bb_mid_s.iloc[i])
                            _bb_bw_i  = float(bb_bw_s.iloc[i])
                            if not np.isnan(_bb_mid_i) and cv < _bb_mid_i:
                                reasons.append("BB Discount")
                                score += 1
                            if not np.isnan(_bb_bw_i) and _bb_bw_i < self.bb_squeeze_pct:
                                reasons.append("BB Squeeze")
                                score += 1
                        _in_premium = (not np.isnan(swing_hi) and swing_hi > swing_lo
                                       and cv >= swing_lo + (swing_hi - swing_lo) * self.discount_pct)
                        if not _in_premium and not _w1_in_premium:
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
                # Sniper mode: build short FVG regardless of H4 bias
                if (htf_bias == -1 or not self.h4_bias_gate) and not self.long_only:
                    bear_gap = l2 - hv
                    if bear_gap >= self.min_fvg_atr * atr_val:
                        ob_lo, ob_hi = _find_bearish_ob(open_, close, high, low, i - 2, self.ob_lookback)
                        is_model3 = ob_lo is not None and min(ob_hi, l2) - max(ob_lo, hv) > 0
                        reasons = ["FVG"]
                        score = 0
                        if ob_lo is not None:
                            reasons.append("OB")
                            score += 1
                        if self.use_bb and bb_mid_s is not None:
                            _bb_mid_i = float(bb_mid_s.iloc[i])
                            _bb_bw_i  = float(bb_bw_s.iloc[i])
                            if not np.isnan(_bb_mid_i) and cv > _bb_mid_i:
                                reasons.append("BB Premium")
                                score += 1
                            if not np.isnan(_bb_bw_i) and _bb_bw_i < self.bb_squeeze_pct:
                                reasons.append("BB Squeeze")
                                score += 1
                        _in_discount = (not np.isnan(swing_hi) and swing_hi > swing_lo
                                        and cv <= swing_lo + (swing_hi - swing_lo) * self.discount_pct)
                        if not _in_discount and not _w1_in_discount:
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

                # ── Alt setups: liquidity sweep reversal + BOS ────────────────
                # Fire when no clean FVG exists but structure/liquidity confirms.
                # Pushed into active_fvgs with tested=True for fast entry.
                _alt_cdh_i = _cdh[i]; _alt_cdl_i = _cdl[i]
                _alt_pdh_i = _pdh[i]; _alt_pdl_i = _pdl[i]

                if htf_bias == 1 or not self.h4_bias_gate:
                    # ── Sweep reversal long ───────────────────────────────────
                    _swept_lo, _sl_lo = _sweep_reversal_bull(close, low, i, self.liq_lookback * 2)
                    if _swept_lo is not None and not _w1_in_premium and not _alt_dup(active_fvgs, "bull", _sl_lo, atr_val):
                        active_fvgs.append({
                            "dir":      "bull",
                            "fvg_lo":   _sl_lo,
                            "fvg_hi":   _swept_lo + 0.3 * atr_val,
                            "fvg_ce":   _swept_lo + 0.15 * atr_val,
                            "ob_lo":    _sl_lo,
                            "ob_hi":    None,
                            "score":    0,
                            "reasons":  ["Sweep reversal"],
                            "formed":   i,
                            "tested":   True,
                            "test_bar": i,
                            "model3":   False,
                            "trend_s":  trend_strength,
                        })

                    # ── BOS long ──────────────────────────────────────────────
                    _bos_level, _bos_hit = _bos_bull(high, close, i, self.ob_lookback)
                    if _bos_hit and _bos_level is not None and not _w1_in_premium and not _alt_dup(active_fvgs, "bull", _bos_level, atr_val):
                        active_fvgs.append({
                            "dir":      "bull",
                            "fvg_lo":   _bos_level,
                            "fvg_hi":   _bos_level + 0.5 * atr_val,
                            "fvg_ce":   _bos_level + 0.25 * atr_val,
                            "ob_lo":    float(low.iloc[i]),
                            "ob_hi":    None,
                            "score":    0,
                            "reasons":  ["BOS"],
                            "formed":   i,
                            "tested":   True,
                            "test_bar": i,
                            "model3":   False,
                            "trend_s":  trend_strength,
                        })

                if (htf_bias == -1 or not self.h4_bias_gate) and not self.long_only:
                    # ── Sweep reversal short ──────────────────────────────────
                    _swept_hi, _sl_hi = _sweep_reversal_bear(close, high, i, self.liq_lookback * 2)
                    if _swept_hi is not None and not _w1_in_discount and not _alt_dup(active_fvgs, "bear", _sl_hi, atr_val):
                        active_fvgs.append({
                            "dir":      "bear",
                            "fvg_lo":   _swept_hi - 0.3 * atr_val,
                            "fvg_hi":   _sl_hi,
                            "fvg_ce":   _swept_hi - 0.15 * atr_val,
                            "ob_lo":    None,
                            "ob_hi":    _sl_hi,
                            "score":    0,
                            "reasons":  ["Sweep reversal"],
                            "formed":   i,
                            "tested":   True,
                            "test_bar": i,
                            "model3":   False,
                            "trend_s":  trend_strength,
                        })

                    # ── BOS short ─────────────────────────────────────────────
                    _bos_level, _bos_hit = _bos_bear(low, close, i, self.ob_lookback)
                    if _bos_hit and _bos_level is not None and not _w1_in_discount and not _alt_dup(active_fvgs, "bear", _bos_level - 0.5 * atr_val, atr_val):
                        active_fvgs.append({
                            "dir":      "bear",
                            "fvg_lo":   _bos_level - 0.5 * atr_val,
                            "fvg_hi":   _bos_level,
                            "fvg_ce":   _bos_level - 0.25 * atr_val,
                            "ob_lo":    None,
                            "ob_hi":    float(high.iloc[i]),
                            "score":    0,
                            "reasons":  ["BOS"],
                            "formed":   i,
                            "tested":   True,
                            "test_bar": i,
                            "model3":   False,
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
                                if _needs_session_gate(self._symbol) and not _active_session(self._symbol, hour):
                                    to_remove.append(fvg); continue
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
                                if _needs_session_gate(self._symbol) and not _active_session(self._symbol, hour):
                                    to_remove.append(fvg); continue
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
