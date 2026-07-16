"""Per-trade agent — individual adjudication of every candidate trade.

Every trade is analyzed on its own and produces a durable TradeTicket: the
liquidity plan (structural stop, liquidity target, type, grade, size), the gate
results, the relevant Council members' voices, and a GO / NO-GO verdict. On close
the ticket is completed with the post-trade review, closing the learning loop.

This is the governance surface Anton asked for: not a silent arithmetic decision,
but a per-trade thesis with the Council on record and dissent captured.

Council voices are deterministic rule-based inspectors of the plan + gates (a real
gate, not a formality). Any VETO blocks the trade. Tickets persist to
logs/trade_tickets.jsonl.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from execution.trade_analyzer import analyze_entry, analyze_exit, TradePlan

logger = logging.getLogger(__name__)

TICKETS = Path("logs") / "trade_tickets.jsonl"


@dataclass
class CouncilVoice:
    member: str
    vote:   str          # "go" | "veto" | "note"
    note:   str


@dataclass
class TradeTicket:
    # identity
    symbol:      str
    direction:   int
    created:     str
    # plan (from analyzer)
    trade_type:  str
    grade:       str
    entry:       float
    stop:        float
    target:      float
    rr:          float
    size_mult:   float
    stop_src:    str
    target_src:  str
    thesis:      str
    # adjudication
    gates:       dict
    council:     list         # list[CouncilVoice]
    verdict:     str          # "GO" | "NO_GO"
    dissent:     list = field(default_factory=list)
    # completion (post-trade)
    closed:      bool = False
    exit_price:  Optional[float] = None
    exit_R:      Optional[float] = None
    hit_target:  Optional[bool] = None
    mfe_R:       Optional[float] = None
    mae_R:       Optional[float] = None
    lesson:      Optional[str] = None


# ── Council voices (deterministic per-trade inspectors) ───────────────────────

def _council_review(plan: TradePlan, gates: dict, dd_room_daily: float,
                    dd_room_total: float) -> list[CouncilVoice]:
    voices: list[CouncilVoice] = []

    # #05 Compliance — is there DD room for this trade?
    if dd_room_daily <= 0 or dd_room_total <= 0:
        voices.append(CouncilVoice("05 Compliance", "veto",
            f"No DD room (daily {dd_room_daily:.2f}%, total {dd_room_total:.2f}%)"))
    else:
        voices.append(CouncilVoice("05 Compliance", "go",
            f"DD room ok (daily {dd_room_daily:.2f}%, total {dd_room_total:.2f}%)"))

    # #06 App Engineer — grade / size sanity
    if plan.grade == "C" or not plan.tradeable:
        voices.append(CouncilVoice("06 App", "veto",
            f"grade {plan.grade}, not tradeable — {plan.thesis}"))
    else:
        voices.append(CouncilVoice("06 App", "go",
            f"grade {plan.grade}, size {plan.size_mult}x"))

    # #08 Performance — entry timing (M5 trigger confirmed?)
    if gates.get("m5_trigger") is False:
        voices.append(CouncilVoice("08 Perf", "veto", "M5 trigger not confirmed"))
    else:
        voices.append(CouncilVoice("08 Perf", "note", "entry timing ok"))

    # #11 Reality Gap — does the drawn liquidity actually exist?
    if plan.target_src == "atr_fallback" or plan.stop_src == "atr_fallback":
        voices.append(CouncilVoice("11 Reality Gap", "veto",
            f"no real structure (stop {plan.stop_src}, target {plan.target_src})"))
    else:
        voices.append(CouncilVoice("11 Reality Gap", "go",
            f"structural: stop@{plan.stop_src} target@{plan.target_src}"))

    # #12 Devil's Advocate — always names the biggest risk; vetoes thin RR
    if plan.rr < 1.2:
        voices.append(CouncilVoice("12 Devil", "veto",
            f"RR {plan.rr} too thin — draw too close to justify risk"))
    else:
        biggest = ("sweep can fail" if plan.trade_type == "sweep_reversal"
                   else "H4 could flip" if plan.trade_type == "breakout"
                   else "trend could stall at the pool")
        voices.append(CouncilVoice("12 Devil", "note", f"risk: {biggest}"))

    return voices


class TradeAgent:
    """Adjudicates one candidate trade end-to-end."""

    def __init__(self, persist: bool = True):
        self._persist = persist
        self._open: dict[str, TradeTicket] = {}

    def evaluate(
        self,
        symbol:    str,
        direction: int,
        df_m15,
        df_m5,
        entry:     float,
        ref_stop:  float,
        atr:       float,
        h4_bias:   int = 0,
        gates:     Optional[dict] = None,
        dd_room_daily: float = 5.0,
        dd_room_total: float = 10.0,
        rr_fallback: float = 2.0,
        swept:     bool = False,
        plan:      Optional[TradePlan] = None,
    ) -> TradeTicket:
        gates = gates or {}
        if plan is None:
            plan = analyze_entry(
                df_m15=df_m15, df_m5=df_m5, direction=direction, entry=entry,
                stop=ref_stop, atr=atr, h4_bias=h4_bias, rr_fallback=rr_fallback, swept=swept,
            )
        voices = _council_review(plan, gates, dd_room_daily, dd_room_total)
        vetoes = [v for v in voices if v.vote == "veto"]
        # hard gate failures also veto
        gate_fail = [k for k, v in gates.items() if v is False]

        verdict = "GO" if (plan.tradeable and not vetoes and not gate_fail) else "NO_GO"
        dissent = [f"{v.member}: {v.note}" for v in vetoes] + \
                  [f"gate:{g} failed" for g in gate_fail]

        ticket = TradeTicket(
            symbol=symbol, direction=direction,
            created=datetime.now(tz=timezone.utc).isoformat(),
            trade_type=plan.trade_type, grade=plan.grade, entry=entry,
            stop=plan.stop, target=plan.tp, rr=plan.rr, size_mult=plan.size_mult,
            stop_src=plan.stop_src, target_src=plan.target_src, thesis=plan.thesis,
            gates=gates, council=[asdict(v) for v in voices],
            verdict=verdict, dissent=dissent,
        )
        logger.info("[TradeAgent] %s %s -> %s [%s/%s RR=%.2f]%s",
                    symbol, "BUY" if direction == 1 else "SELL", verdict,
                    plan.trade_type, plan.grade, plan.rr,
                    (" | dissent: " + "; ".join(dissent)) if dissent else "")
        if verdict == "GO":
            self._open[symbol] = ticket
        if self._persist:
            self._write(ticket)
        return ticket

    def complete(
        self, symbol: str, exit_price: float,
        path_high: float, path_low: float, reason: str,
    ) -> Optional[TradeTicket]:
        t = self._open.pop(symbol, None)
        if t is None:
            return None
        review = analyze_exit(
            direction=t.direction, entry=t.entry, stop=t.stop, target=t.target,
            exit_px=exit_price, path_high=path_high, path_low=path_low, reason=reason,
        )
        t.closed     = True
        t.exit_price = exit_price
        t.exit_R     = review.exit_R
        t.hit_target = review.hit_target
        t.mfe_R      = review.mfe_R
        t.mae_R      = review.mae_R
        t.lesson     = review.lesson
        logger.info("[TradeAgent] CLOSE %s exitR=%.2f hitTP=%s | %s",
                    symbol, review.exit_R, review.hit_target, review.lesson)
        if self._persist:
            self._write(t)
        return t

    def _write(self, t: TradeTicket) -> None:
        try:
            TICKETS.parent.mkdir(parents=True, exist_ok=True)
            with open(TICKETS, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(t)) + "\n")
        except Exception as exc:
            logger.warning("[TradeAgent] ticket write failed: %s", exc)
