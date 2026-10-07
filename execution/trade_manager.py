"""Adaptive trade manager — probability engine for open position decisions.

Runs every bar on every open position. Aggregates multi-timeframe signal weights
into counter/continuation scores, estimates sweep risk, builds a probability tree,
and outputs a TradeAction with inline council adjudication notes.

Weighting:
  M5 signals → full weight (1.0); counter weight discounted when H4 aligned with trade
  M1 signals → half weight (0.5) — influence only, never sole trigger
  H4 + News  → applied directly (not TF-weighted)

Action thresholds (tunable via TradeManager constructor):
  counter ≥ EXIT_THRESH   AND sweep_risk ≤ sweep_thresh → EXIT
  counter ≥ EXIT_THRESH   AND sweep_risk > sweep_thresh  → WAIT (one-bar confirmation)
  counter ≥ PARTIAL_THRESH                               → PARTIAL_CLOSE + TIGHTEN_SL
  counter ≥ TIGHTEN_THRESH                               → TIGHTEN_SL to nearest structure
  cont ≥ EXTEND_THRESH    AND R ≥ min_extend_r           → EXTEND_TP to next structure
  cont ≥ RUNNER_THRESH    (and no counter action)        → HOLD_RUNNER (tighten SL, TP open)
  else                                                    → HOLD

ADD action is Phase 2 — requires multi-position tracking and adaptive RR
validation in backtest before enabling on live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from execution.signal_detectors import (
    detect_choch,
    detect_bos,
    detect_counter_fvg,
    detect_momentum_shift,
    detect_hh_ll,
    detect_htf_continuation,
    find_structure_sl,
    find_next_structure_tp,
    detect_sweep_recovery,
)


# ── Action types ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    HOLD          = "HOLD"
    TIGHTEN_SL    = "TIGHTEN_SL"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    EXTEND_TP     = "EXTEND_TP"
    EXIT          = "EXIT"
    WAIT          = "WAIT"          # sweep_risk high — hold one bar before exiting
    HOLD_RUNNER   = "HOLD_RUNNER"   # cont ≥ runner_thresh — tighten SL, leave TP open
    ADD           = "ADD"           # Phase 2


# ── Position state ────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    """Snapshot of the open position passed to TradeManager.evaluate()."""
    direction:     int             # +1 long, -1 short
    entry_price:   float
    initial_sl:    float           # never moves — used as R denominator
    current_sl:    float           # live SL (may have been trailed)
    current_tp:    float
    current_price: float           # live mid-price
    bars_elapsed:  int
    t1_hit:           bool = False    # T1 partial already fired
    h4_bias:          int  = 0        # +1, -1, 0 — from strategy or external H4 compute
    consecutive_waits: int = 0        # incremented each time WAIT fires; reset on any other action
    peak_r:           float = 0.0     # best R reached this trade (orchestrator-tracked)

    @property
    def risk_dist(self) -> float:
        return abs(self.entry_price - self.initial_sl)

    @property
    def current_r(self) -> float:
        """Unrealized P&L in R-multiples. Positive = profit, negative = loss."""
        if self.risk_dist <= 0:
            return 0.0
        if self.direction == 1:
            return (self.current_price - self.entry_price) / self.risk_dist
        return (self.entry_price - self.current_price) / self.risk_dist


# ── Trade action ──────────────────────────────────────────────────────────────

@dataclass
class TradeAction:
    action:        ActionType
    new_sl:        Optional[float] = None   # TIGHTEN_SL / PARTIAL_CLOSE / HOLD_RUNNER / EXTEND_TP
    new_tp:        Optional[float] = None   # EXTEND_TP
    close_pct:     Optional[float] = None   # PARTIAL_CLOSE fraction (0–1)
    counter_score: int   = 0
    cont_score:    int   = 0
    reason:        str   = ""
    probability:   float = 1.0              # confidence in primary action (0–1)
    alternatives:  list  = field(default_factory=list)  # [{action, prob, reason}, ...]
    sweep_risk:    float = 0.0              # 0–1 sweep probability
    council_notes: list  = field(default_factory=list)  # [str, ...] from Council adjudication

    def is_structural_change(self) -> bool:
        return self.action in (
            ActionType.TIGHTEN_SL, ActionType.EXTEND_TP,
            ActionType.PARTIAL_CLOSE, ActionType.EXIT, ActionType.HOLD_RUNNER,
        )


# ── Trade manager ─────────────────────────────────────────────────────────────

class TradeManager:
    """Adaptive multi-timeframe trade manager with probability tree and sweep detection."""

    def __init__(
        self,
        exit_thresh:      int   = 3,     # was 4 — with h4_ctr_discount=0.85, max counter≈3.4
        partial_thresh:   int   = 2,     # was 3 — fires on CHoCH+BOS or CHoCH+FVG
        tighten_thresh:   int   = 1,     # was 2 — any single counter signal tightens SL
        extend_thresh:    int   = 3,
        runner_thresh:    int   = 2,
        partial_pct:      float = 0.25,  # was 0.30 — smaller per-event so ladder can repeat
        m5_weight:        float = 1.0,
        m1_weight:        float = 0.5,
        min_extend_r:     float = 1.0,
        atr_period:       int   = 14,
        sweep_thresh:     float = 0.45,
        h4_ctr_discount:  float = 0.85,  # was 0.70 — 4 signals × 0.85 = 3.4 → exit reaches 3
    ):
        self._exit_thresh     = exit_thresh
        self._partial_thresh  = partial_thresh
        self._tighten_thresh  = tighten_thresh
        self._extend_thresh   = extend_thresh
        self._runner_thresh   = runner_thresh
        self._partial_pct     = partial_pct
        self._m5_w            = m5_weight
        self._m1_w            = m1_weight
        self._min_extend_r    = min_extend_r
        self._atr_period      = atr_period
        self._sweep_thresh    = sweep_thresh
        self._h4_ctr_discount = h4_ctr_discount

    # ── Main evaluation loop ──────────────────────────────────────────────────

    def evaluate(
        self,
        position:           PositionState,
        df_m15:             pd.DataFrame,
        df_m5:              Optional[pd.DataFrame] = None,
        df_m1:              Optional[pd.DataFrame] = None,
        news_confirmed_dir: int = 0,
        portfolio_pnl_r:    float = 0.0,
    ) -> TradeAction:
        """Evaluate all signals and return a single concrete action for this bar."""
        counter      = 0.0
        continuation = 0.0
        reasons: list[str] = []

        pos_dir    = position.direction
        atr_m5     = self._compute_atr(df_m5) if df_m5 is not None else 0.0
        h4_aligned = (position.h4_bias == pos_dir)

        # When H4 aligns with our trade, M5 counter signals are less trustworthy
        # (more likely sweeps into aligned trend, not genuine reversals)
        m5_ctr_w = self._m5_w * (self._h4_ctr_discount if h4_aligned else 1.0)

        # ── M5 signals ────────────────────────────────────────────────────────
        if df_m5 is not None and len(df_m5) >= 25:
            v = detect_choch(df_m5, pos_dir, atr=atr_m5)
            if v:
                counter += v * m5_ctr_w
                reasons.append(f"M5_CHoCH:{v * m5_ctr_w:.1f}")

            v = detect_bos(df_m5, pos_dir)
            if v:
                counter += v * m5_ctr_w
                reasons.append(f"M5_BOS:{v * m5_ctr_w:.1f}")

            v = detect_counter_fvg(df_m5, pos_dir, atr=atr_m5)
            if v:
                counter += v * m5_ctr_w
                reasons.append(f"M5_FVG:{v * m5_ctr_w:.1f}")

            v = detect_momentum_shift(df_m5, pos_dir)
            if v:
                counter += v * m5_ctr_w
                reasons.append(f"M5_MOM:{v * m5_ctr_w:.1f}")

            v = detect_hh_ll(df_m5, pos_dir)
            if v:
                continuation += v * self._m5_w
                reasons.append(f"M5_HHLL:{v}")

        # ── M1 signals (half weight) ──────────────────────────────────────────
        if df_m1 is not None and len(df_m1) >= 25:
            m1_ctr_w = self._m1_w * (self._h4_ctr_discount if h4_aligned else 1.0)

            v = detect_choch(df_m1, pos_dir)
            if v:
                counter += v * m1_ctr_w
                reasons.append(f"M1_CHoCH:{v * m1_ctr_w:.1f}")

            v = detect_counter_fvg(df_m1, pos_dir)
            if v:
                counter += v * m1_ctr_w
                reasons.append(f"M1_FVG:{v * m1_ctr_w:.1f}")

            v = detect_hh_ll(df_m1, pos_dir)
            if v:
                continuation += v * self._m1_w
                reasons.append(f"M1_HHLL:{v * self._m1_w:.1f}")

            v = detect_momentum_shift(df_m1, pos_dir)
            if v:
                counter += v * m1_ctr_w
                reasons.append(f"M1_MOM:{v * m1_ctr_w:.1f}")

        # ── HTF bias ─────────────────────────────────────────────────────────
        v = detect_htf_continuation(position.h4_bias, pos_dir)
        if v:
            continuation += v
            reasons.append(f"H4_aligned:{v}")
        elif position.h4_bias != 0 and position.h4_bias != pos_dir:
            counter += 1.0
            reasons.append("H4_opposed:1")

        # ── News ─────────────────────────────────────────────────────────────
        if news_confirmed_dir != 0:
            if news_confirmed_dir == pos_dir:
                continuation += 1.0
                reasons.append("news_aligned:1")
            else:
                # Apply H4 discount to news opposition same as M5 counter signals —
                # a news event against the trade while H4 is aligned is often a spike,
                # not a genuine reversal signal, and should not solo-trigger an exit.
                news_ctr_w = self._h4_ctr_discount if h4_aligned else 1.0
                counter += 2.0 * news_ctr_w
                reasons.append(f"news_opposed:{2.0 * news_ctr_w:.1f}")

        c  = int(round(counter))
        co = int(round(continuation))

        # ── Sweep detection ───────────────────────────────────────────────────
        sweep_risk = 0.0
        if df_m5 is not None and len(df_m5) >= 15:
            sweep_risk = detect_sweep_recovery(df_m5, pos_dir, atr=atr_m5)

        # ── Portfolio P&L gate ────────────────────────────────────────────────
        # When the portfolio is winning overall and this position is losing,
        # exit sooner to protect accumulated gains.
        cur_r = position.current_r
        if portfolio_pnl_r >= 1.0 and cur_r < -0.3:
            # Portfolio up 1R+, this trade losing — cut it now, don't wait for counter
            c = max(c, self._exit_thresh)
            reasons.append(f"portfolio_gate(pnl={portfolio_pnl_r:.1f}R pos={cur_r:.1f}R):force_exit")
        elif portfolio_pnl_r >= 0.5 and cur_r < -0.2:
            # Portfolio up 0.5R, trade losing — log but don't force counter;
            # automatic SL tightening from a 0.5R portfolio edge is too aggressive
            # and stops out normal retracements before the trade has room to work.
            reasons.append(f"portfolio_gate(pnl={portfolio_pnl_r:.1f}R pos={cur_r:.1f}R):noted")

        reason_str = " | ".join(reasons) if reasons else "no_signals"
        action = self._select_action(
            position, df_m5, df_m15, atr_m5, c, co, sweep_risk, reason_str
        )
        return self._apply_profit_lock_floor(position, action)

    @staticmethod
    def _profit_lock_r(peak_r: float) -> Optional[float]:
        if peak_r >= 2.0:
            return 1.0
        if peak_r >= 1.2:
            return 0.5
        if peak_r >= 0.6:
            return 0.05
        return None

    def _apply_profit_lock_floor(self, position: PositionState, action: TradeAction) -> TradeAction:
        # TIGHTEN_SL / EXTEND_TP / HOLD_RUNNER branches return before the ladder
        # in _select_action, so a structural SL below the lock level used to win
        # (a 0.9R trade got SL 99.75 instead of BE+0.05R). The floor is enforced
        # here so no SL-moving branch can place a stop worse than the lock.
        if action.new_sl is None or position.risk_dist <= 0:
            return action
        lock_r = self._profit_lock_r(max(position.peak_r, position.current_r))
        if lock_r is None:
            return action
        d = position.direction
        lock_px = position.entry_price + d * lock_r * position.risk_dist
        worse = (d == 1 and action.new_sl < lock_px) or (d == -1 and action.new_sl > lock_px)
        # A stop on the wrong side of price is rejected by the broker; leave the
        # action unchanged rather than send an invalid modify.
        valid = (d == 1 and lock_px < position.current_price) or (d == -1 and lock_px > position.current_price)
        if worse and valid:
            action.new_sl = lock_px
            action.reason = f"{action.reason} | PROFIT_LOCK_FLOOR +{lock_r:.2f}R"
        return action

    # ── Action selection ──────────────────────────────────────────────────────

    def _select_action(
        self,
        position:   PositionState,
        df_m5:      Optional[pd.DataFrame],
        df_m15:     pd.DataFrame,
        atr_m5:     float,
        counter:    int,
        cont:       int,
        sweep_risk: float,
        reason:     str,
    ) -> TradeAction:
        pos_dir    = position.direction
        df_primary = df_m5 if df_m5 is not None else df_m15
        cur_r      = position.current_r

        probs = self._compute_action_probabilities(counter, cont, sweep_risk, cur_r)

        # ── Minimum hold guard ────────────────────────────────────────────────
        # Never exit within the first 3 bars (45 min on M15) unless the trade
        # is already in loss beyond 0.5R — early counter signals are almost
        # always noise before the move has had room to develop.
        MIN_HOLD_BARS = 3
        early_exit_blocked = (
            position.bars_elapsed < MIN_HOLD_BARS
            and cur_r > -0.5
        )

        # ── EXIT / WAIT ───────────────────────────────────────────────────────
        if counter >= self._exit_thresh and not early_exit_blocked:
            if sweep_risk > self._sweep_thresh:
                # 3+ consecutive WAITs = sweep isn't clearing; cap exposure via PARTIAL_CLOSE
                if position.consecutive_waits >= 3:
                    new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
                    action = ActionType.PARTIAL_CLOSE
                    return TradeAction(
                        action        = action,
                        close_pct     = self._partial_pct,
                        new_sl        = new_sl,
                        counter_score = counter,
                        cont_score    = cont,
                        reason        = f"PARTIAL(sweep_stuck {position.consecutive_waits}W) counter={counter} | {reason}",
                        probability   = probs.get(ActionType.PARTIAL_CLOSE, 0.5),
                        alternatives  = self._top_alternatives(probs, ActionType.PARTIAL_CLOSE),
                        sweep_risk    = sweep_risk,
                        council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
                    )
                # Stop-hunt sweep likely — wait one bar for confirmation before exiting
                action = ActionType.WAIT
                return TradeAction(
                    action        = action,
                    counter_score = counter,
                    cont_score    = cont,
                    reason        = f"WAIT sweep_risk={sweep_risk:.2f} counter={counter} waits={position.consecutive_waits} | {reason}",
                    probability   = probs.get(action, 0.5),
                    alternatives  = self._top_alternatives(probs, action),
                    sweep_risk    = sweep_risk,
                    council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
                )
            else:
                action = ActionType.EXIT
                return TradeAction(
                    action        = action,
                    counter_score = counter,
                    cont_score    = cont,
                    reason        = f"EXIT counter={counter} | {reason}",
                    probability   = probs.get(action, 0.7),
                    alternatives  = self._top_alternatives(probs, action),
                    sweep_risk    = sweep_risk,
                    council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
                )

        # ── PARTIAL_CLOSE: significant counter, reduce + tighten ──────────────
        if counter >= self._partial_thresh and not early_exit_blocked:
            new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
            action = ActionType.PARTIAL_CLOSE
            return TradeAction(
                action        = action,
                close_pct     = self._partial_pct,
                new_sl        = new_sl,
                counter_score = counter,
                cont_score    = cont,
                reason        = f"PARTIAL({self._partial_pct:.0%}) counter={counter} | {reason}",
                probability   = probs.get(action, 0.5),
                alternatives  = self._top_alternatives(probs, action),
                sweep_risk    = sweep_risk,
                council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
            )

        # ── RUNNER TRIM: T1 already hit, any counter signal = reduce runner ────
        # After the first partial at T1, the runner should be protected more
        # aggressively — a single counter signal is enough to trim it further.
        if position.t1_hit and counter >= 2 and not early_exit_blocked and cur_r >= 0.5:
            new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
            action = ActionType.PARTIAL_CLOSE
            return TradeAction(
                action        = action,
                close_pct     = 0.25,
                new_sl        = new_sl,
                counter_score = counter,
                cont_score    = cont,
                reason        = f"RUNNER_TRIM(25%) t1_hit counter={counter} R={cur_r:.2f} | {reason}",
                probability   = probs.get(ActionType.PARTIAL_CLOSE, 0.5),
                alternatives  = self._top_alternatives(probs, ActionType.PARTIAL_CLOSE),
                sweep_risk    = sweep_risk,
                council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
            )

        # ── TIGHTEN_SL: mild counter, move SL to structure ────────────────────
        if counter >= self._tighten_thresh:
            new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
            if new_sl is not None:
                action = ActionType.TIGHTEN_SL
                return TradeAction(
                    action        = action,
                    new_sl        = new_sl,
                    counter_score = counter,
                    cont_score    = cont,
                    reason        = f"TIGHTEN_SL→{new_sl:.5f} counter={counter} | {reason}",
                    probability   = probs.get(action, 0.4),
                    alternatives  = self._top_alternatives(probs, action),
                    sweep_risk    = sweep_risk,
                    council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
                )

        # ── EXTEND_TP: strong continuation, push TP to next structure ─────────
        if cont >= self._extend_thresh and cur_r >= self._min_extend_r:
            min_ext = position.risk_dist * 0.5
            new_tp  = find_next_structure_tp(
                df_m15, pos_dir, position.current_tp, min_extension=min_ext
            )
            if new_tp is not None:
                # Also tighten SL when extending TP to lock in progress
                new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
                action = ActionType.EXTEND_TP
                return TradeAction(
                    action        = action,
                    new_tp        = new_tp,
                    new_sl        = new_sl,
                    counter_score = counter,
                    cont_score    = cont,
                    reason        = f"EXTEND_TP→{new_tp:.5f} cont={cont} R={cur_r:.2f} | {reason}",
                    probability   = probs.get(action, 0.5),
                    alternatives  = self._top_alternatives(probs, action),
                    sweep_risk    = sweep_risk,
                    council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
                )

        # ── HOLD_RUNNER: early continuation — tighten SL, keep TP open ────────
        if cont >= self._runner_thresh:
            new_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
            action = ActionType.HOLD_RUNNER
            return TradeAction(
                action        = action,
                new_sl        = new_sl,
                counter_score = counter,
                cont_score    = cont,
                reason        = f"HOLD_RUNNER cont={cont} R={cur_r:.2f} | {reason}",
                probability   = probs.get(action, 0.4),
                alternatives  = self._top_alternatives(probs, action),
                sweep_risk    = sweep_risk,
                council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
            )

        # ── Profit-lock ladder ────────────────────────────────────────────────
        # Live 2026-07-27: winners at 0.5–0.9R round-tripped to full stops
        # (JP225 gave back 1.5R). Once a trade has worked, the stop follows —
        # regardless of counter/continuation signals. Never loosens.
        peak = max(position.peak_r, cur_r)
        lock_r = None
        if peak >= 2.0:
            lock_r = 1.0
        elif peak >= 1.2:
            lock_r = 0.5
        elif peak >= 0.6:
            lock_r = 0.05   # breakeven + spread buffer
        if lock_r is not None and position.risk_dist > 0:
            lock_px = position.entry_price + pos_dir * lock_r * position.risk_dist
            # Prefer structural SL to avoid placing stop inside swing noise;
            # only use arithmetic lock_px as the minimum floor.
            struct_sl = find_structure_sl(df_primary, pos_dir, position.current_sl, atr=atr_m5)
            if struct_sl is not None:
                satisfies = ((pos_dir == 1 and struct_sl >= lock_px) or
                             (pos_dir == -1 and struct_sl <= lock_px))
                if satisfies:
                    lock_px = struct_sl
            improves = ((pos_dir == 1 and lock_px > position.current_sl) or
                        (pos_dir == -1 and lock_px < position.current_sl))
            if improves:
                action = ActionType.TIGHTEN_SL
                return TradeAction(
                    action        = action,
                    new_sl        = lock_px,
                    counter_score = counter,
                    cont_score    = cont,
                    reason        = f"PROFIT_LOCK peak={peak:.2f}R → SL locks +{lock_r:.2f}R | {reason}",
                    probability   = 0.9,
                    alternatives  = self._top_alternatives(probs, action),
                    sweep_risk    = sweep_risk,
                    council_notes = [f"[C07 SRE] Trade reached {peak:.2f}R — stop ratcheted to "
                                     f"+{lock_r:.2f}R; a worked trade must never round-trip to a full loss."],
                )

        # ── HOLD ─────────────────────────────────────────────────────────────
        action = ActionType.HOLD
        return TradeAction(
            action        = action,
            counter_score = counter,
            cont_score    = cont,
            reason        = f"HOLD c={counter} co={cont} | {reason}",
            probability   = probs.get(action, 0.5),
            alternatives  = self._top_alternatives(probs, action),
            sweep_risk    = sweep_risk,
            council_notes = self._council_notes(action, counter, cont, sweep_risk, position, probs),
        )

    # ── Probability tree ──────────────────────────────────────────────────────

    def _compute_action_probabilities(
        self,
        counter:    int,
        cont:       int,
        sweep_risk: float,
        cur_r:      float,
    ) -> dict:
        """
        Probability mass for each possible action. Values sum to ~1.0.
        Reflects signal strength relative to thresholds and sweep context.
        """
        et = max(self._exit_thresh, 1)
        pt = max(self._partial_thresh, 1)
        tt = max(self._tighten_thresh, 1)
        rt = max(self._runner_thresh, 1)
        xt = max(self._extend_thresh, 1)

        exit_raw  = max(0.0, counter / et)
        wait_w    = exit_raw * sweep_risk
        exit_w    = exit_raw * (1.0 - sweep_risk * 0.8)
        partial_w = max(0.0, counter / pt) * (1.0 if counter < et else 0.3)
        tighten_w = max(0.0, counter / tt) * (1.0 if counter < pt else 0.2)
        runner_w  = max(0.0, cont / rt) * (1.0 if cont >= rt else 0.0)
        extend_w  = max(0.0, cont / xt) * (1.0 if cur_r >= self._min_extend_r else 0.2)
        dominant  = max(exit_raw, runner_w, extend_w)
        hold_w    = max(0.05, 0.4 * (1.0 - dominant))

        raw = {
            ActionType.EXIT:          max(0.0, exit_w),
            ActionType.WAIT:          max(0.0, wait_w),
            ActionType.PARTIAL_CLOSE: max(0.0, partial_w),
            ActionType.TIGHTEN_SL:    max(0.0, tighten_w),
            ActionType.HOLD_RUNNER:   max(0.0, runner_w),
            ActionType.EXTEND_TP:     max(0.0, extend_w),
            ActionType.HOLD:          hold_w,
        }
        total = sum(raw.values())
        if total <= 0:
            return {ActionType.HOLD: 1.0}
        return {k: round(v / total, 3) for k, v in raw.items() if v > 0.01}

    def _top_alternatives(
        self, probs: dict, primary: ActionType, n: int = 3
    ) -> list:
        return [
            {"action": k.value, "prob": v}
            for k, v in sorted(probs.items(), key=lambda x: -x[1])
            if k != primary and v >= 0.05
        ][:n]

    # ── Council adjudication ──────────────────────────────────────────────────

    def _council_notes(
        self,
        action:     ActionType,
        counter:    int,
        cont:       int,
        sweep_risk: float,
        position:   PositionState,
        probs:      dict,
    ) -> list[str]:
        """Inline council perspectives. C11 (Reality Gap) and C12 (Devil's Advocate) always fire."""
        notes = []
        cur_r = position.current_r

        # ── C11: Reality Gap Analyst — always ────────────────────────────────
        if action == ActionType.EXIT:
            notes.append(
                f"[C11 Reality Gap] EXIT on bar-close prices; live fills worse. "
                f"sweep_risk={sweep_risk:.2f} — if sweep was genuine, live tick data "
                f"may not confirm counter={counter} at bar close."
            )
        elif action == ActionType.WAIT:
            notes.append(
                f"[C11 Reality Gap] WAIT over EXIT (sweep_risk={sweep_risk:.2f}). "
                "Live tick data resolves sweeps faster than M5 bars. "
                "One-bar WAIT adds slippage cost if sweep extends into trend."
            )
        elif action == ActionType.HOLD_RUNNER:
            notes.append(
                f"[C11 Reality Gap] HOLD_RUNNER at cont={cont}, R={cur_r:.2f}. "
                "Structural SL from M5 swings — live spread may stop out before "
                "the model's swing low."
            )
        elif action == ActionType.EXTEND_TP:
            notes.append(
                f"[C11 Reality Gap] EXTEND_TP at cont={cont}, R={cur_r:.2f}. "
                "M15 structural TP can gap through in fast news moves. "
                "Backtest assumes clean fills at structural price."
            )
        elif action == ActionType.PARTIAL_CLOSE:
            notes.append(
                f"[C11 Reality Gap] PARTIAL at counter={counter}. Remaining position "
                f"still open; sweep_risk={sweep_risk:.2f} may indicate this is a sweep "
                "that will reverse — reducing exposure is correct defensively."
            )
        else:
            notes.append(
                f"[C11 Reality Gap] {action.value}: c={counter} co={cont} R={cur_r:.2f}. "
                "No action — confirm no feed divergence between model and live price."
            )

        # ── Conditional members ───────────────────────────────────────────────
        if action == ActionType.EXIT and cur_r < 0:
            notes.append(
                "[C05 Compliance] Exit at negative R — check daily drawdown exposure "
                "before executing on FTMO. Must stay within 5% daily loss limit."
            )

        if action == ActionType.WAIT and sweep_risk > 0.7:
            notes.append(
                f"[C06 App Engineer] sweep_risk={sweep_risk:.2f} is high — WAIT is correct. "
                "Add max-consecutive-WAIT guard: if 3+ consecutive WAITs fire, "
                "downgrade to PARTIAL_CLOSE to cap exposure."
            )

        if action in (ActionType.HOLD_RUNNER, ActionType.EXTEND_TP):
            notes.append(
                "[C03 Data Engineer] SL update must persist correctly — verify "
                "current_sl is not overwritten by parallel trail-stop logic. "
                "Both modifications must be applied in the correct order."
            )

        if action == ActionType.WAIT:
            notes.append(
                "[C07 SRE] WAIT: position stays open — confirm heartbeat is active "
                "and next-bar evaluation will fire. Stale connection = no exit."
            )

        # ── C12: Devil's Advocate — always last ──────────────────────────────
        if action == ActionType.EXIT:
            notes.append(
                f"[C12 Devil's Advocate] counter={counter} — is this 4 independent "
                "signals or BOS(3) + noise(1)? PARTIAL at "
                f"{self._partial_thresh} reduces exposure without full exit. "
                "Confirm at least 2 signal types contributed."
            )
        elif action == ActionType.WAIT:
            notes.append(
                f"[C12 Devil's Advocate] WAIT is right at sweep_risk={sweep_risk:.2f}. "
                "But next bar confirming counter signals means you exit at worse fills. "
                "Track WAIT→EXIT chains — do they outperform direct EXIT in live?"
            )
        elif action == ActionType.HOLD_RUNNER:
            notes.append(
                f"[C12 Devil's Advocate] HOLD_RUNNER at cont={cont} — threshold is "
                f"low ({self._runner_thresh}). H4_aligned(2)+HHLL(2) alone hits cont=4. "
                "Is this structural continuation or HTF bias + sideways consolidation?"
            )
        elif action == ActionType.EXTEND_TP:
            notes.append(
                f"[C12 Devil's Advocate] EXTEND_TP at R={cur_r:.2f} — are you letting "
                "a winner run, or giving back profit chasing a level? Verify the next "
                "structural target is not a major HTF resistance zone."
            )
        elif action == ActionType.PARTIAL_CLOSE:
            notes.append(
                f"[C12 Devil's Advocate] PARTIAL at counter={counter} — correct response "
                "to counter pressure. Confirm new_sl is genuinely tighter, not None "
                "(no structural level found = no protection added)."
            )
        else:
            notes.append(
                f"[C12 Devil's Advocate] HOLD at c={counter} co={cont} — verify TM "
                "thresholds are calibrated to live signal frequency. If thresholds are "
                "never triggered, TM is a passenger, not a manager."
            )

        return notes

    # ── ATR helper ────────────────────────────────────────────────────────────

    def _compute_atr(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < self._atr_period + 1:
            return 0.0
        high  = df["high"]
        low   = df["low"]
        prev  = df["close"].shift(1)
        tr    = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return float(tr.ewm(span=self._atr_period, adjust=False).mean().iloc[-1])
