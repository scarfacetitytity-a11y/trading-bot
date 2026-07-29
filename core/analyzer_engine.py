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

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent.parent / "logs" / "analyzer_v2.jsonl"
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

    # Minimum stack score for a candidate to be BORN (placeholder — calibrate from Phase 1 logs)
    _BORN_THRESHOLD = 35.0

    def __init__(self, symbol: str, strategy) -> None:
        self.symbol   = symbol
        self._strategy = strategy
        self._reader   = MarketReader(symbol)
        self._slm      = SignalLifecycleManager(symbol, max_watch_bars=8)
        self._bar_idx  = 0
        self._state    = AnalyzerState(symbol=symbol)
        self._lock     = threading.Lock()
        self._profile  = PROFILES.get(symbol)

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

    def update(self, df: pd.DataFrame, df_m5: Optional[pd.DataFrame] = None) -> None:
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

            # 2. Get strategy signal for this bar
            desired = 0
            signal_score = 0
            try:
                desired      = int(self._strategy.generate_signal(df))
                _sc          = getattr(self._strategy, "_scores", None)
                signal_score = int(_sc.iloc[-1]) if _sc is not None else 0
            except Exception:
                pass

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
            if desired != 0 and signal_score > 0 and self._profile:
                inp = StackInput(
                    symbol=self.symbol,
                    signal_dir=desired,
                    archetype=self._profile.archetype,
                    archetype_threshold=self._profile.entry_threshold,
                    bar_time=bar_time,
                    base_score=signal_score,
                    m5_confirmed=(desired in m5_confirmed_dirs),
                    h4_aligned=(getattr(self._strategy, "_last_h4_bias", 0) == desired),
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
            }
