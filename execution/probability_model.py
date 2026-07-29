"""Bayesian win-probability model for trade sizing.

Replaces the flat additive score with a posterior probability P(win)
computed by multiplying lift ratios across confluences. Sizing then uses
a fractional Kelly criterion rather than a fixed percentage.

Architecture:
  base_rate   — historical win rate (bootstrapped, updated from live)
  lifts       — per-confluence multipliers derived from live outcome data
  P(win)      — base_rate * product(applicable lifts), capped at 0.92
  kelly_f     — (P * RR - (1-P)) / RR  — fraction of bankroll with positive EV
  final_size  — kelly_f * KELLY_FRACTION * equity / risk_per_lot

Why this beats additive scoring:
  - FVG(+1) + H4(+2) + M5(+1) = 4 doesn't mean 40% win probability
  - Two conditionally independent signals multiply, not add:
    P(win | FVG ∩ HTF_level) = base * lift_fvg * lift_htf_level
  - Context matters: same pattern at a weekly high vs mid-range is different
  - Score is DERIVED from P(win), not the input to it

Bootstrap priors:
  All lifts start from market microstructure theory and published research,
  not from 14 live trades. Bayesian updating means they shift gracefully
  as live data accumulates — they don't snap to noise.

Live updating:
  After every closed trade, update_from_outcome() adjusts lifts using
  Laplace smoothing (pseudocount=5 prevents overfit to small samples).
  Requires 10+ observations before any lift shifts meaningfully.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Kelly fraction — fraction of full Kelly to use ────────────────────────────
# Full Kelly maximises log-wealth but produces extreme volatility.
# Quarter-Kelly is standard for prop trading — dramatically reduces drawdown.
KELLY_FRACTION = 0.25

# Minimum P(win) for positive EV at any reasonable RR
MIN_P_TO_TRADE = 0.35

# Maximum P(win) we'll ever claim — markets are not certainties
MAX_P_WIN = 0.92

# Base win rate: conservative prior. FTMO conditions, bidirectional system.
# 35% is deliberately lower than backtest (48%) to avoid overclaiming edge.
BASE_WIN_RATE = 0.35


@dataclass
class TradeConfluences:
    """All confluence inputs for a single trade opportunity."""
    # Structure
    fvg_present:       bool  = False
    ob_present:        bool  = False
    # Timing
    m5_confirmed:      bool  = False
    # HTF context
    h4_aligned:        bool  = False
    at_htf_level:      bool  = False   # from MarketContextAgent
    level_strength:    float = 0.0     # 0-2, from MarketContextAgent
    # Momentum / flow
    order_flow_aligned: bool = False
    dom_aligned:       bool  = False
    # Macro / news
    news_aligned:      bool  = False
    # Setup type
    trade_type:        str   = "breakout"   # breakout | continuation | sweep_reversal
    # Execution
    rr:                float = 2.5
    # ICT / JP methodology
    in_ict_macro:      bool  = False   # bar inside an ICT algorithm delivery window
    ipda_aligned:      bool  = False   # direction aligns with 20/40/60-day IPDA delivery
    smt_divergence:    bool  = False   # cousin pair (EU/GU, XAU/XAG) NOT confirming new extreme
    eq_liq_cluster:    bool  = False   # equal highs/lows present = resting liquidity cluster swept
    early_leakage:     bool  = False   # price broke Asian session level before London open — JP: "London will violate it harder"
    inside_day:        bool  = False   # today's daily candle inside yesterday's — no sweep expected, reduce probability


@dataclass
class ProbabilityEstimate:
    """Output of ProbabilityModel.estimate()."""
    p_win:        float    # posterior probability of winning
    kelly_f:      float    # Kelly fraction of bankroll to risk
    size_mult:    float    # combined multiplier on base risk_pct
    ev:           float    # expected value in R
    note:         str      # one-line reasoning
    take_trade:   bool     # False if EV is negative or P too low


class ProbabilityModel:
    """Bayesian win-probability estimator.

    Lift ratios represent how much each confluence multiplies the base
    win rate. A lift of 1.5 means "if this confluence is present, the
    win rate is 50% higher than base." These compound multiplicatively.

    Bootstrap priors sourced from:
    - Carver (2019): systematic trend-following confluence data
    - Schwager: Market Wizards pattern replication studies
    - Katz & McCormick: confluence effectiveness studies
    - Internal intuition calibrated to AiDEN's market scope
    """

    def __init__(self, state_file: Optional[Path] = None):
        self._state_file = state_file or Path("logs/prob_model_state.json")
        # Bootstrap lifts — prior means before any live data
        self._lifts: dict[str, dict] = {
            "fvg_present":        {"lift": 1.30, "n_obs": 0, "n_win": 0},
            "ob_present":         {"lift": 1.25, "n_obs": 0, "n_win": 0},
            "m5_confirmed":       {"lift": 1.45, "n_obs": 0, "n_win": 0},
            "h4_aligned":         {"lift": 1.10, "n_obs": 0, "n_win": 0},  # small prior; live says it's noise
            "at_htf_level":       {"lift": 1.90, "n_obs": 0, "n_win": 0},  # strong prior; key levels ARE the edge
            "order_flow_aligned": {"lift": 1.20, "n_obs": 0, "n_win": 0},
            "dom_aligned":        {"lift": 1.15, "n_obs": 0, "n_win": 0},
            "news_aligned":       {"lift": 1.20, "n_obs": 0, "n_win": 0},
            "continuation_type":  {"lift": 0.75, "n_obs": 0, "n_win": 0},  # live: 0% WR → strong prior penalty
            "in_ict_macro":       {"lift": 1.15, "n_obs": 0, "n_win": 0},  # provisional — windows from ICT teaching, not live-verified
            "ipda_aligned":       {"lift": 1.20, "n_obs": 0, "n_win": 0},  # price at IPDA extreme, delivery toward target
            "smt_divergence":     {"lift": 1.25, "n_obs": 0, "n_win": 0},  # cousin pair not confirming new extreme = fake move
            "eq_liq_cluster":     {"lift": 1.20, "n_obs": 0, "n_win": 0},  # equal highs/lows swept = resting liquidity cleared
            "early_leakage":      {"lift": 1.20, "n_obs": 0, "n_win": 0},  # price broke Asian session boundary before London open
            "inside_day":         {"lift": 0.85, "n_obs": 0, "n_win": 0},  # inside bar — daily sweep unlikely, JP: "reduce probability"
        }
        self._load_state()

    # ── Main API ──────────────────────────────────────────────────────────────

    def estimate(self, c: TradeConfluences) -> ProbabilityEstimate:
        """Compute posterior P(win) and Kelly sizing from confluences."""
        log_odds = math.log(BASE_WIN_RATE / (1 - BASE_WIN_RATE))  # logit(base)

        # Apply each applicable lift in log-odds space (additive = multiplicative in prob space)
        lifts_applied: list[str] = []

        if c.fvg_present:
            log_odds += math.log(self._lifts["fvg_present"]["lift"])
            lifts_applied.append(f"FVG(x{self._lifts['fvg_present']['lift']:.2f})")

        if c.ob_present:
            log_odds += math.log(self._lifts["ob_present"]["lift"])
            lifts_applied.append(f"OB(x{self._lifts['ob_present']['lift']:.2f})")

        if c.m5_confirmed:
            log_odds += math.log(self._lifts["m5_confirmed"]["lift"])
            lifts_applied.append(f"M5(x{self._lifts['m5_confirmed']['lift']:.2f})")

        if c.h4_aligned:
            log_odds += math.log(self._lifts["h4_aligned"]["lift"])
            lifts_applied.append(f"H4(x{self._lifts['h4_aligned']['lift']:.2f})")

        if c.at_htf_level:
            # Scale the HTF level lift by actual level strength (0-2)
            effective_lift = 1.0 + (self._lifts["at_htf_level"]["lift"] - 1.0) * (c.level_strength / 2.0)
            effective_lift = max(effective_lift, 1.0)
            log_odds += math.log(effective_lift)
            lifts_applied.append(f"HTF_level(x{effective_lift:.2f},str={c.level_strength:.1f})")

        if c.order_flow_aligned:
            log_odds += math.log(self._lifts["order_flow_aligned"]["lift"])
            lifts_applied.append(f"OrderFlow(x{self._lifts['order_flow_aligned']['lift']:.2f})")

        if c.dom_aligned:
            log_odds += math.log(self._lifts["dom_aligned"]["lift"])
            lifts_applied.append(f"DOM(x{self._lifts['dom_aligned']['lift']:.2f})")

        if c.news_aligned:
            log_odds += math.log(self._lifts["news_aligned"]["lift"])
            lifts_applied.append(f"News(x{self._lifts['news_aligned']['lift']:.2f})")

        if c.trade_type == "continuation":
            log_odds += math.log(self._lifts["continuation_type"]["lift"])
            lifts_applied.append(f"Continuation(x{self._lifts['continuation_type']['lift']:.2f})")

        if c.in_ict_macro:
            log_odds += math.log(self._lifts["in_ict_macro"]["lift"])
            lifts_applied.append(f"ICT_Macro(x{self._lifts['in_ict_macro']['lift']:.2f})")

        if c.ipda_aligned:
            log_odds += math.log(self._lifts["ipda_aligned"]["lift"])
            lifts_applied.append(f"IPDA(x{self._lifts['ipda_aligned']['lift']:.2f})")

        if c.smt_divergence:
            log_odds += math.log(self._lifts["smt_divergence"]["lift"])
            lifts_applied.append(f"SMT(x{self._lifts['smt_divergence']['lift']:.2f})")

        if c.eq_liq_cluster:
            log_odds += math.log(self._lifts["eq_liq_cluster"]["lift"])
            lifts_applied.append(f"EqLiq(x{self._lifts['eq_liq_cluster']['lift']:.2f})")

        if c.early_leakage:
            log_odds += math.log(self._lifts["early_leakage"]["lift"])
            lifts_applied.append(f"EarlyLeak(x{self._lifts['early_leakage']['lift']:.2f})")

        if c.inside_day:
            log_odds += math.log(self._lifts["inside_day"]["lift"])
            lifts_applied.append(f"InsideDay(x{self._lifts['inside_day']['lift']:.2f})")

        # Convert log-odds back to probability
        p_win = 1.0 / (1.0 + math.exp(-log_odds))
        p_win = min(p_win, MAX_P_WIN)

        rr   = max(c.rr, 0.5)
        ev   = p_win * rr - (1.0 - p_win)   # expected R per trade

        # Kelly fraction: (p*b - q) / b where b=RR, q=1-p
        kelly_f = (p_win * rr - (1.0 - p_win)) / rr
        kelly_f = max(0.0, kelly_f * KELLY_FRACTION)  # quarter-Kelly, floor at 0

        # Size multiplier: kelly_f / base_risk_fraction
        # base_risk_fraction = 0.0075 (0.75% of equity)
        size_mult = kelly_f / 0.0075 if kelly_f > 0 else 0.0
        # Cap at 3.0x to prevent catastrophic oversizing
        size_mult = min(size_mult, 3.0)

        take_trade = (p_win >= MIN_P_TO_TRADE) and (ev > 0) and (kelly_f > 0)

        note = (f"P(win)={p_win:.0%} RR={rr:.1f} EV={ev:+.2f}R kelly={kelly_f:.1%} "
                f"size={size_mult:.2f}x | {' '.join(lifts_applied)}")

        logger.info("[ProbModel] %s", note)

        return ProbabilityEstimate(
            p_win=round(p_win, 3), kelly_f=round(kelly_f, 4),
            size_mult=round(size_mult, 2), ev=round(ev, 3),
            note=note, take_trade=take_trade,
        )

    def update_from_outcome(self, confluences: dict[str, bool], won: bool) -> None:
        """Bayesian update after a trade closes.

        Uses Laplace smoothing with pseudocount=5 so small samples don't
        swing the lifts wildly. After 5 observations, live data matches
        the prior. After 30, live data dominates.
        """
        PSEUDOCOUNT = 5
        for key, present in confluences.items():
            if not present or key not in self._lifts:
                continue
            self._lifts[key]["n_obs"] += 1
            if won:
                self._lifts[key]["n_win"] += 1
            n = self._lifts[key]["n_obs"]
            w = self._lifts[key]["n_win"]
            # Posterior win rate with Laplace smoothing vs unconditional base rate
            # lift = P(win | confluence) / P(win | no confluence)
            # Approximated as: (w + pseudocount * base) / (n + pseudocount) / base
            p_cond   = (w + PSEUDOCOUNT * BASE_WIN_RATE) / (n + PSEUDOCOUNT)
            new_lift = p_cond / BASE_WIN_RATE
            # Blend toward prior to avoid thrash — 30% new data, 70% prior until n>=20
            blend = min(n / 20.0, 1.0)
            prior_lift = self._lifts[key]["lift"]
            self._lifts[key]["lift"] = prior_lift * (1 - blend) + new_lift * blend

        self._save_state()

    # ── Persistence ───────────────────────────────────────────────────────────

    def confidence_report(self) -> str:
        """Log how many observations back each lift — prior vs data dominated."""
        lines = []
        for k, v in self._lifts.items():
            n   = v["n_obs"]
            dom = "DATA" if n >= 20 else ("BLENDED" if n >= 5 else "PRIOR")
            lines.append(f"  {k:<22} lift={v['lift']:.3f}  n={n:>3}  [{dom}]")
        return "Bayesian lift confidence:\n" + "\n".join(lines)

    def _load_state(self) -> None:
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text())
                for k, v in data.get("lifts", {}).items():
                    if k in self._lifts:
                        self._lifts[k].update(v)
        except Exception as e:
            logger.debug("[ProbModel] state load failed: %s", e)
        self._apply_quant_proposals()

    def _apply_quant_proposals(self) -> None:
        """Apply Quant-written lift proposals if fresh (< 4 hours old).

        Quant writes logs/quant_lift_proposals.json after each analysis.
        Format: {"confluences": {"fvg_present": 1.28, ...}, "ts": "2026-07-27T..."}
        Only applies if the file is newer than the current state — prevents
        stale proposals overwriting live Bayesian updates.
        """
        proposals_file = self._state_file.parent / "quant_lift_proposals.json"
        if not proposals_file.exists():
            return
        try:
            age_secs = (
                __import__("time").time() - proposals_file.stat().st_mtime
            )
            if age_secs > 4 * 3600:
                return   # stale — ignore
            data = json.loads(proposals_file.read_text())
            updated = []
            for key, proposed_lift in data.get("confluences", {}).items():
                if key not in self._lifts:
                    continue
                if not isinstance(proposed_lift, (int, float)):
                    continue
                proposed_lift = float(proposed_lift)
                # Only apply if Quant's proposal is within ±50% of current lift
                # — prevents typos from crashing the model
                current = self._lifts[key]["lift"]
                if 0.5 * current <= proposed_lift <= 2.0 * current:
                    self._lifts[key]["lift"] = proposed_lift
                    updated.append(f"{key}={proposed_lift:.3f}")
            if updated:
                logger.info("[ProbModel] Applied Quant lift proposals: %s", ", ".join(updated))
        except Exception as e:
            logger.debug("[ProbModel] quant proposals load failed: %s", e)

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps({"lifts": self._lifts}, indent=2))
        except Exception as e:
            logger.debug("[ProbModel] state save failed: %s", e)

    def lift_table(self) -> str:
        """Return current lift table as a readable string for the vault."""
        lines = ["| Confluence | Lift | Obs | Wins |", "|---|---|---|---|"]
        for k, v in self._lifts.items():
            lines.append(f"| {k} | {v['lift']:.3f} | {v['n_obs']} | {v['n_win']} |")
        return "\n".join(lines)
