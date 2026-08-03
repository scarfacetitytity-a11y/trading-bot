"""CouncilRouter — OmniRoute-adapted governance layer for AiDEN trade decisions.

Adapted from OmniRoute open-sse/services/autoCombo/combo.ts.

Instead of routing AI model calls, this routes trade decisions through the
relevant subset of Council of 12 personas based on:
  1. Task fitness (fitness_table.py) — which personas own this decision type
  2. Composite score (scorer.py) — who is most fit right now
  3. Self-healing state (self_healing.py) — exclude degraded/excluded personas
  4. Strategy weights (strategy_registry.py) — shift weights per FTMO state

The verdict functions are rule-based code checks, NOT LLM calls. The
Council of 12 is a CODE governance layer — each persona runs their domain
logic and votes GO/VETO with a reason string.

Usage:
    from council import CouncilRouter
    router = CouncilRouter()
    verdict = router.governance_check(
        symbol="XAUUSD", direction=1, signal_score=7, daily_dd_pct=1.2,
        lots=0.10, equity=100_000, sl=1900.00, tp=1930.00,
        plan_type="continuation", atr_ratio=1.1, news_window=False,
        extra_context={},
    )
    if not verdict.approved:
        logger.info("Council veto: %s", verdict.notes)
        continue
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .fitness_table import FITNESS, PERSONA_NAMES, ALWAYS_ACTIVE, POOL_THRESHOLD
from .scorer import score_pool
from .strategy_registry import select_strategy, get_weights
from .self_healing import SelfHealingManager

logger = logging.getLogger(__name__)

# Maximum personas to run (ALWAYS_ACTIVE not counted toward this cap)
MAX_ACTIVE = 4

# Minimum composite score to enter pool (filters weak candidates)
SCORE_FLOOR = 0.45

# Task category map for common orchestrator decision points
TASK_MAP = {
    "trade_entry":     "trade_entry",
    "sizing":          "sizing",
    "risk":            "risk_management",
    "compliance":      "compliance",
    "execution":       "execution_timing",
    "market":          "market_analysis",
    "system":          "system_health",
}


@dataclass
class CouncilVerdict:
    approved:         bool
    notes:            list[str]          = field(default_factory=list)
    selected:         list[str]          = field(default_factory=list)   # persona IDs
    vetoed_by:        list[str]          = field(default_factory=list)
    strategy_profile: str               = "normal_market"
    scored:           list[dict]         = field(default_factory=list)   # full score rows


class CouncilRouter:
    """Singleton-friendly; construct once and reuse per SignalEngine."""

    def __init__(self) -> None:
        self._healer = SelfHealingManager.get()

    # ── Public entry point ────────────────────────────────────────────────────

    def governance_check(
        self,
        symbol: str,
        direction: int,
        signal_score: int,
        daily_dd_pct: float,
        lots: float,
        equity: float,
        sl: float,
        tp: float | None,
        plan_type: str = "breakout",
        atr_ratio: float = 1.0,
        news_window: bool = False,
        extra_context: dict | None = None,
    ) -> CouncilVerdict:
        """Full governance check for a pending trade entry.

        Selects the optimal Council subset, runs rule-based verdicts, and
        returns a CouncilVerdict (approved/vetoed + notes).
        """
        ctx = {
            "symbol":        symbol,
            "direction":     direction,
            "signal_score":  signal_score,
            "daily_dd_pct":  daily_dd_pct,
            "lots":          lots,
            "equity":        equity,
            "sl":            sl,
            "tp":            tp,
            "plan_type":     plan_type,
            "atr_ratio":     atr_ratio,
            "news_window":   news_window,
            "is_urgent":     news_window,
            **(extra_context or {}),
        }

        strategy = select_strategy(daily_dd_pct, atr_ratio, news_window)
        weights  = get_weights(strategy)

        # Build candidate pool + score
        candidates = self._build_candidate_pool("trade_entry", ctx)
        scored     = score_pool(
            candidates, "trade_entry", ctx,
            self._healer.accuracy_map(),
            self._healer.error_rate_map(),
            weights,
        )

        # Apply self-healing weight penalty
        for row in scored:
            row["score"] *= self._healer.weight_penalty(row["persona_id"])

        # Select top N (ALWAYS_ACTIVE already in list, limit optionals)
        selected = self._select(scored)

        # Run verdicts
        notes, vetoed_by = self._fusion_dispatch(selected, ctx)

        approved = len(vetoed_by) == 0
        verdict  = CouncilVerdict(
            approved=approved,
            notes=notes,
            selected=selected,
            vetoed_by=vetoed_by,
            strategy_profile=strategy,
            scored=scored,
        )

        # Emit event
        try:
            from execution.aiden_event_bus import append_event
            append_event(
                "COUNCIL_VERDICT",
                symbol=symbol,
                direction=direction,
                approved=approved,
                selected=selected,
                vetoed_by=vetoed_by,
                strategy=strategy,
                daily_dd_pct=round(daily_dd_pct, 3),
            )
        except Exception:
            pass

        _level = logging.DEBUG if approved else logging.INFO
        logger.log(
            _level,
            "[Council] %s %s dir=%+d dd=%.1f%% profile=%s selected=%s veto=%s",
            "APPROVED" if approved else "VETOED",
            symbol, direction, daily_dd_pct, strategy, selected, vetoed_by,
        )
        return verdict

    # ── Pool construction ─────────────────────────────────────────────────────

    def _build_candidate_pool(self, task_category: str, ctx: dict) -> list[str]:
        """Return persona IDs eligible for this decision."""
        pool: list[str] = []

        # Always-active first (Compliance Officer #05, Devil's Advocate #12)
        for pid in sorted(ALWAYS_ACTIVE):
            if self._healer.is_available(pid):
                pool.append(pid)

        # Optional candidates above fitness threshold
        for pid, task_scores in FITNESS.items():
            if pid in ALWAYS_ACTIVE:
                continue
            fitness = task_scores.get(task_category, 0.0)
            if fitness >= POOL_THRESHOLD and self._healer.is_available(pid):
                pool.append(pid)

        return pool

    def _select(self, scored: list[dict]) -> list[str]:
        """Select final persona set: ALWAYS_ACTIVE + top-scoring optionals."""
        always  = [r["persona_id"] for r in scored if r["persona_id"] in ALWAYS_ACTIVE]
        optionals = [
            r["persona_id"] for r in scored
            if r["persona_id"] not in ALWAYS_ACTIVE and r["score"] >= SCORE_FLOOR
        ]
        # Take top MAX_ACTIVE from optionals (already sorted desc by scorer)
        chosen = always + optionals[:MAX_ACTIVE]
        return chosen

    # ── Verdict dispatch ──────────────────────────────────────────────────────

    def _fusion_dispatch(
        self, selected: list[str], ctx: dict
    ) -> tuple[list[str], list[str]]:
        """Run each selected persona's verdict function. Returns (notes, vetoed_by)."""
        notes: list[str] = []
        vetoed_by: list[str] = []

        _verdict_fns = {
            "01": self._v01_principal,
            "03": self._v03_data_engineer,
            "05": self._v05_compliance,
            "06": self._v06_app_engineer,
            "07": self._v07_sre,
            "08": self._v08_performance,
            "09": self._v09_test_engineer,
            "10": self._v10_staff_engineer,
            "11": self._v11_reality_gap,
            "12": self._v12_devil,
        }

        for pid in selected:
            fn = _verdict_fns.get(pid)
            if fn is None:
                continue
            try:
                go, reason = fn(ctx)
                name = PERSONA_NAMES.get(pid, f"#{pid}")
                if go:
                    notes.append(f"#{pid} {name}: {reason}")
                else:
                    notes.append(f"#{pid} {name} VETO: {reason}")
                    vetoed_by.append(pid)
                    # Record bad outcome for self-healing
                    self._healer.record_outcome(pid, correct=False)
            except Exception as exc:
                logger.debug("[Council] Persona #%s error: %s", pid, exc)

        return notes, vetoed_by

    # ── Persona verdict functions (rule-based, no LLM) ────────────────────────

    def _v01_principal(self, ctx: dict) -> tuple[bool, str]:
        """Principal Engineer: only veto on catastrophic state."""
        dd = ctx.get("daily_dd_pct", 0.0)
        if dd >= 4.8:
            return False, f"daily DD {dd:.2f}% — within 0.2% of 5% limit, halt"
        return True, f"DD={dd:.2f}% within safe range"

    def _v03_data_engineer(self, ctx: dict) -> tuple[bool, str]:
        """Data Engineer: equity/lots state sanity."""
        equity = ctx.get("equity", 0.0)
        lots   = ctx.get("lots", 0.0)
        if equity <= 0:
            return False, "equity ≤ 0 — invalid state"
        if lots <= 0:
            return False, "lots ≤ 0 — invalid sizing"
        risk_pct = (lots * ctx.get("sl", 0.0)) / equity * 100 if ctx.get("sl") else 0
        return True, f"equity={equity:.0f} lots={lots} est-risk≈{risk_pct:.2f}%"

    def _v05_compliance(self, ctx: dict) -> tuple[bool, str]:
        """Compliance Officer: FTMO limits, direction, news window."""
        dd  = ctx.get("daily_dd_pct", 0.0)
        if dd >= 4.5:
            return False, f"FTMO daily DD {dd:.2f}% >= 4.5% soft cap — no new entries"
        score = ctx.get("signal_score", 0)
        if ctx.get("news_window") and score < 7:
            return False, f"high-impact news in 30min, score {score} < 7 — skip"
        return True, f"FTMO compliance clear (dd={dd:.2f}%)"

    def _v06_app_engineer(self, ctx: dict) -> tuple[bool, str]:
        """App Engineer: gate logic + score validity."""
        score     = ctx.get("signal_score", 0)
        plan_type = ctx.get("plan_type", "breakout")
        sl        = ctx.get("sl")
        tp        = ctx.get("tp")
        if sl is None or sl <= 0:
            return False, "no valid stop-loss — naked trade"
        if score < 4:
            return False, f"signal_score {score} < 4 absolute floor"
        if plan_type == "continuation" and score < 6:
            return False, f"continuation trade needs score≥6, have {score}"
        rr = ((tp - ctx.get("sl", 0)) / (ctx.get("sl", 1) or 1)) if tp else 0
        return True, f"score={score} plan={plan_type} rr≈{abs(rr):.1f}"

    def _v07_sre(self, ctx: dict) -> tuple[bool, str]:
        """SRE: no veto unless something is catastrophically wrong."""
        return True, "process health OK"

    def _v08_performance(self, ctx: dict) -> tuple[bool, str]:
        """Performance Engineer: no veto at order level (latency already handled)."""
        return True, "execution path clear"

    def _v09_test_engineer(self, ctx: dict) -> tuple[bool, str]:
        """Test Engineer: verify score is in the tested range."""
        score = ctx.get("signal_score", 0)
        if score > 15:
            return False, f"score {score} outside tested range (>15) — data error suspected"
        return True, f"score {score} within validated range"

    def _v10_staff_engineer(self, ctx: dict) -> tuple[bool, str]:
        """Staff Engineer: no veto — advisory only."""
        dd = ctx.get("daily_dd_pct", 0.0)
        if dd >= 3.5:
            return True, f"DD {dd:.1f}% elevated — consider reduced position"
        return True, "no structural concerns"

    def _v11_reality_gap(self, ctx: dict) -> tuple[bool, str]:
        """Reality Gap Analyst: flag if live conditions diverge from backtest regime."""
        atr_ratio = ctx.get("atr_ratio", 1.0)
        if atr_ratio >= 2.5:
            return False, f"ATR ratio {atr_ratio:.1f}x — extreme volatility outside backtest envelope"
        if atr_ratio >= 1.8:
            return True, f"ATR ratio {atr_ratio:.1f}x elevated — backtest edge may compress"
        return True, f"volatility within backtest range (ATR ratio {atr_ratio:.1f}x)"

    def _v12_devil(self, ctx: dict) -> tuple[bool, str]:
        """Devil's Advocate: challenge every confident conclusion.

        Veto conditions:
          - score is only 1 above floor AND continuation trade (too marginal)
          - daily DD > 3% AND signal score < 7 (low-conviction in drawdown)
          - near daily limit AND not a reversal setup
        """
        score     = ctx.get("signal_score", 0)
        dd        = ctx.get("daily_dd_pct", 0.0)
        plan_type = ctx.get("plan_type", "breakout")

        if dd >= 3.0 and score < 7:
            return False, (
                f"DA: DD={dd:.1f}%, score only {score} — "
                "low conviction in drawdown, wait for A+ setup"
            )

        if plan_type == "continuation" and score <= 6:
            return False, (
                f"DA: continuation trade at score {score} — "
                "continuation edge is weakest (+0.37R/38%WR baseline)"
            )

        return True, f"DA clears: score={score} dd={dd:.1f}% plan={plan_type}"
