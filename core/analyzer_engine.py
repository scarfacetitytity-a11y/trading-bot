"""Phase 3: AnalyzerEngine — v3 shadow dry-run engine.

Runs alongside each live TradingEngine, processing the same M15 bars with
the v3 logic stack (MarketReader + SignalLifecycle + ProbabilityStack).
Logs all decisions to logs/analyzer_v2.jsonl. No live trades ever placed.

Phase 4: When shadow logs show the v3 stack outperforms the gate cascade
(higher would_have_fired accuracy, fewer false blocks), AnalyzerEngine
replaces TradingEngine per symbol in the staged cutover sequence.

Heartbeat: registered with the orchestrator's HeartbeatRegistry so it
appears in the dashboard like any other component.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.market_reader import MarketReader, BarEvent, ZoneState, ZoneType, Zone
from core.signal_lifecycle import SignalLifecycleManager, SignalCandidate, SignalState
from core.probability_stack import StackInput, compute_stack_score
from core.instrument_profile import PROFILES
from execution.scenario_engine import ScenarioEngine

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent.parent / "logs" / "analyzer_v2.jsonl"

# ICT algorithm delivery macro windows (UTC minutes from midnight).
# Mirrors orchestrator._in_ict_macro — kept here to avoid circular import.
_ICT_WINDOWS = [
    (2*60+33, 3*60),
    (4*60+3,  4*60+30),
    (8*60+50, 9*60+10),
    (9*60+50, 10*60+10),
    (10*60+50, 11*60+10),
    (13*60+10, 13*60+40),
    (14*60+50, 15*60+10),
    (15*60+15, 15*60+45),
    (19*60+55, 20*60+5),
]


def _in_ict_macro_window(bar_time: str) -> bool:
    try:
        dt = pd.Timestamp(bar_time)
        t = dt.hour * 60 + dt.minute
        return any(s <= t <= e for s, e in _ICT_WINDOWS)
    except Exception:
        return False
_LOG_LOCK = threading.Lock()


def _write_log(row: dict) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


@dataclass
class AnalyzerState:
    symbol:      str
    bar_idx:     int = 0
    last_bar:    Optional[str] = None
    live_zones:  int = 0
    armed_count: int = 0
    fired_count: int = 0
    blocked_count: int = 0
    total_candidates: int = 0


class AnalyzerEngine:
    """V3 shadow engine for one symbol.

    Processes M15 bars via update(df, df_m5). Owns a MarketReader and
    SignalLifecycleManager. Does not interact with MT5 or place any order.
    """

    # Score gate disabled — ProbabilityStack is analysis-only, not an entry gate.
    # All strategy signals are born; confluence data collected for quant analysis.
    _BORN_THRESHOLD = 0.0

    def __init__(self, symbol: str, strategy, news_gate=None) -> None:
        self.symbol   = symbol
        self._strategy = strategy
        self._news_gate = news_gate
        self._reader   = MarketReader(symbol)
        self._slm      = SignalLifecycleManager(symbol, max_watch_bars=8)
        self._bar_idx  = 0
        self._state    = AnalyzerState(symbol=symbol)
        self._lock     = threading.Lock()
        self._profile  = PROFILES.get(symbol)
        self._scenario_engine = ScenarioEngine()

        # Heartbeat support (compatible with orchestrator HeartbeatRegistry)
        self._last_beat: float = time.time()
        self._beat_interval: float = 60.0
        self.name = f"AnalyzerV2[{symbol}]"

    # ── HeartbeatRegistry interface ───────────────────────────────────────────

    def beat(self, msg: str = "") -> None:
        self._last_beat = time.time()

    def seconds_since_beat(self) -> float:
        return time.time() - self._last_beat

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, df: pd.DataFrame, df_m5: Optional[pd.DataFrame] = None, amd_phase=None) -> None:
        """Process one M15 bar. Call this after each bar closes."""
        if df is None or len(df) < 3:
            return

        with self._lock:
            self._bar_idx += 1
            try:
                bar_time = str(pd.to_datetime(df["time"].iloc[-1], utc=True))
            except Exception:
                bar_time = datetime.now(timezone.utc).isoformat()

            # 1. Run MarketReader — detect zones, advance states
            bar_events = self._reader.update(df)
            violated_dirs: set[int] = set()
            for ev in bar_events:
                from core.market_reader import IntraBarEvent
                if ev.event == IntraBarEvent.ZONE_VIOLATED:
                    violated_dirs.add(ev.zone.direction)
                _write_log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "zone_event",
                    "symbol": self.symbol,
                    "bar": bar_time,
                    "event": ev.event.value,
                    "zone_type": ev.zone.zone_type.value,
                    "direction": ev.zone.direction,
                    "price": ev.price,
                    "note": ev.note,
                })

            # AMD phase: derive primary signal direction before strategy runs
            _amd_direction = 0
            if amd_phase is not None:
                try:
                    from core.amd_detector import MarketPhase as _MP
                    if amd_phase.phase in (_MP.MANIPULATION, _MP.DISPLACEMENT):
                        if amd_phase.sweep is not None:
                            _amd_direction = amd_phase.sweep.direction
                            logger.info(
                                "[V3][%s] AMD %s | sweep %s @ %.5g → primary dir %+d",
                                self.symbol, amd_phase.phase.value,
                                "BULL" if _amd_direction == 1 else "BEAR",
                                float(getattr(amd_phase.sweep, "swept_price", 0.0)),
                                _amd_direction,
                            )
                except Exception:
                    pass

            # 2. Get strategy signal for this bar (entry zone data; AMD drives direction)
            _strategy_desired = 0
            signal_score = 0
            try:
                _sigs             = self._strategy.generate_signals(df)
                _strategy_desired = int(_sigs.iloc[-1])
                _sc               = getattr(self._strategy, "_scores", None)
                signal_score      = int(_sc.iloc[-1]) if _sc is not None else 0
            except Exception:
                pass
            _confl = getattr(self._strategy, "_last_confluences", {})

            # AMD direction takes priority; strategy is FVG/OB zone data only
            desired = _amd_direction if _amd_direction != 0 else _strategy_desired

            # News + ICT macro window — computed once per bar, used in StackInput below
            _news_aligned = False
            if self._news_gate is not None and desired != 0:
                try:
                    _nctx = self._news_gate.get_context()
                    _news_aligned = _nctx.score_modifier(self.symbol, desired) > 0
                except Exception:
                    pass
            _in_ict_macro = _in_ict_macro_window(bar_time)

            # 3. M5 confirmation check
            m5_confirmed_dirs: set[int] = set()
            if df_m5 is not None and desired != 0:
                try:
                    from execution.signal_detectors import detect_m5_entry_trigger
                    atr_m5 = float(
                        (df_m5["high"] - df_m5["low"]).rolling(14).mean().iloc[-1]
                    ) if len(df_m5) > 14 else 0.0
                    if detect_m5_entry_trigger(df_m5, desired, atr=atr_m5):
                        m5_confirmed_dirs.add(desired)
                except Exception:
                    pass

            # 4. Compute stack score if signal present
            # AMD narrative is the gate; FVG/OB score gate removed
            if desired != 0 and self._profile:
                # Get H4 bias for scenario evaluation
                _h4 = getattr(self._strategy, "_last_h4_bias", 0)
                try:
                    _h4 = int(self._strategy.current_h4_bias(df)) if hasattr(self._strategy, "current_h4_bias") else _h4
                except Exception:
                    pass

                # Run ScenarioEngine to classify the setup (S1-S4)
                _atr_val = 0.0
                try:
                    _a = getattr(self._strategy, "_atr_cache", None)
                    if _a is not None and len(_a) > 0:
                        import math
                        _v = float(_a.iloc[-1])
                        _atr_val = _v if not math.isnan(_v) else 0.0
                except Exception:
                    pass
                try:
                    _scenario = self._scenario_engine.evaluate(
                        symbol=self.symbol, direction=desired,
                        df_m15=df, df_m5=df_m5, h4_bias=_h4, atr=_atr_val,
                    )
                except Exception:
                    _scenario = None

                # fvg_present: True only if strategy found an FVG in the AMD direction.
                # When AMD drives direction without a matching FVG, checklist score
                # is lower (→ smaller sizing), but trade is not blocked.
                _fvg_aligned = (_strategy_desired == desired and signal_score > 0)
                # sweep_present: True when AMD sweep fired OR strategy detected it
                _sweep_present = (
                    _amd_direction != 0  # AMD phase detected a sweep
                    or _confl.get("sweep_present", False)
                )
                inp = StackInput(
                    symbol=self.symbol,
                    signal_dir=desired,
                    archetype=self._profile.archetype,
                    archetype_threshold=self._profile.entry_threshold,
                    bar_time=bar_time,
                    base_score=signal_score,
                    fvg_present=_fvg_aligned,
                    m5_confirmed=(desired in m5_confirmed_dirs),
                    h4_aligned=(_h4 == desired),
                    ob_present=_confl.get("ob_present", False),
                    sweep_present=_sweep_present,
                    level_strength=_confl.get("level_strength", 0.0),
                    news_aligned=_news_aligned,
                    in_ict_macro=_in_ict_macro,
                    scenario_class=_scenario.scenario if _scenario else "none",
                    scenario_confidence=_scenario.confidence if _scenario else 0.0,
                    scenario_go=(_scenario.verdict in ("GO", "SHADOW_GO")) if _scenario else False,
                )
                stack_score = compute_stack_score(inp)

                # 5. Birth new candidate if score meets floor
                if stack_score >= self._BORN_THRESHOLD:
                    # Check for existing active candidate in this direction
                    existing = [c for c in self._slm.active() if c.direction == desired]
                    if not existing:
                        live_zones = self._reader.get_live_zones(desired)
                        zone_lo = live_zones[-1].price_lo if live_zones else 0.0
                        zone_hi = live_zones[-1].price_hi if live_zones else 0.0
                        zone_type = live_zones[-1].zone_type.value if live_zones else "fvg"
                        cand = SignalCandidate(
                            symbol=self.symbol,
                            direction=desired,
                            born_bar=self._bar_idx,
                            stack_score=stack_score,
                            base_score=signal_score,
                            score_floor=self._profile.entry_threshold,
                            archetype=self._profile.archetype,
                            zone_lo=zone_lo,
                            zone_hi=zone_hi,
                            zone_type=zone_type,
                        )
                        self._slm.add(cand)
                        self._state.total_candidates += 1
                        _write_log({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "type": "signal_born",
                            "symbol": self.symbol,
                            "bar": bar_time,
                            "direction": desired,
                            "stack_score": stack_score,
                            "base_score": signal_score,
                            "archetype": self._profile.archetype,
                            "threshold": self._profile.entry_threshold,
                            "would_have_fired": stack_score >= self._profile.entry_threshold,
                        })

            # 6. Tick lifecycle manager
            armed = self._slm.tick(
                bar_idx=self._bar_idx,
                violated_directions=violated_dirs,
                m5_confirmed_directions=m5_confirmed_dirs,
            )

            # 7. Log armed candidates (would-be entries)
            for cand in armed:
                self._state.armed_count += 1
                _write_log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "signal_armed",
                    "symbol": self.symbol,
                    "bar": bar_time,
                    "candidate": cand.to_dict(),
                    "would_enter": cand.stack_score >= (self._profile.entry_threshold if self._profile else 65),
                })

            # 8. Update state snapshot
            self._state.bar_idx  = self._bar_idx
            self._state.last_bar = bar_time
            self._state.live_zones = len(self._reader.get_live_zones())
            self.beat(f"bar={bar_time[-5:]} | signal={desired:+d} | zones={self._state.live_zones}")

    # ── V3 live mode interface ────────────────────────────────────────────────

    def get_armed_direction(self) -> int:
        """Return the direction (+1/-1) of the highest-scoring ARMED candidate, or 0.

        Threshold gate removed: AMD narrative is the entry gate.
        Sizing is modulated by get_sizing_mult() instead.
        """
        with self._lock:
            armed = [c for c in self._slm.all_candidates() if c.state.value == "armed"]
            if not armed:
                return 0
            best = max(armed, key=lambda c: c.stack_score)
            return best.direction

    def get_armed_score(self) -> float:
        """Return the stack score of the best ARMED candidate, or 0."""
        with self._lock:
            armed = [c for c in self._slm.all_candidates() if c.state.value == "armed"]
            if not armed:
                return 0.0
            return max(c.stack_score for c in armed)

    def get_sizing_mult(self) -> float:
        """Return sizing multiplier 0.6–2.0 based on checklist score.

        Replaces threshold gate: low-context AMD trades fire at 0.6x,
        high-confluence setups scale up to 2.0x. Nothing blocks except FTMO limits.
        """
        with self._lock:
            armed = [c for c in self._slm.all_candidates() if c.state.value == "armed"]
            if not armed:
                return 1.0
            score = max(c.stack_score for c in armed)
            if score >= 70: return 2.0
            if score >= 55: return 1.5
            if score >= 40: return 1.25
            if score >= 25: return 1.0
            if score >= 15: return 0.75
            return 0.6

    def consume_armed(self, direction: int) -> None:
        """Mark ARMED candidates in direction as FIRED (called by TradingEngine on trade fire)."""
        with self._lock:
            self._slm.mark_fired(direction)
            self._state.fired_count += 1

    @staticmethod
    def validate_cutover_ready(symbol: str, shadow_log_path: Path, min_samples: int = 50) -> tuple[bool, str]:
        """Check shadow_stack.jsonl for minimum signal samples before allowing live cutover."""
        try:
            count = 0
            with shadow_log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        if row.get("symbol") == symbol:
                            count += 1
                    except Exception:
                        pass
            if count >= min_samples:
                return True, f"{symbol}: {count} samples ≥ {min_samples} — cutover ready"
            return False, f"{symbol}: only {count}/{min_samples} shadow samples — run longer before cutover"
        except FileNotFoundError:
            return False, f"shadow_stack.jsonl not found — Phase 1 must run first"
        except Exception as e:
            return False, f"validation error: {e}"

    def status(self) -> dict:
        with self._lock:
            active_cands = self._slm.active()
            return {
                "symbol":     self.symbol,
                "bar_idx":    self._state.bar_idx,
                "last_bar":   self._state.last_bar,
                "live_zones": self._state.live_zones,
                "active_candidates": len(active_cands),
                "total_candidates":  self._state.total_candidates,
                "armed_count": self._state.armed_count,
                "fired_count": self._state.fired_count,
            }
