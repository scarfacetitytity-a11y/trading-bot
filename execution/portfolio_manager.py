"""Adaptive portfolio risk allocator.

A cross-position risk manager that replaces passive caps/halts with active
balance. It does four things (behaviours 1-3 live here; dynamic TP — #4 — lives
in TradeManager):

  1. New trades adapt to the remaining daily-risk budget AND their quality —
     a strong setup is never dropped, only sized to fit.
  2. Quality-weighted reallocation — when a new setup is technically stronger
     than an open, weaker/losing position, trim risk from the weaker trade and
     fund the better one (trade-management balance, not a static block).
  3. Aggregate daily-loss guarantee — actively trim/flatten the weakest exposure
     so the book can never run open positions past the daily loss limit.

This module is PURE LOGIC: it takes a snapshot of open positions and returns
*actions* (grant a size, trim these tickets, flatten these). The caller executes
them. That keeps it fully unit-testable without a broker connection.

Gated off by default (config `adaptive_portfolio: false`) until the bar-level
simulation validates it — we do not ship un-backtested trimming into the hot
path unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpenPos:
    """Snapshot of one open bot position."""
    symbol:    str
    score:     int          # entry quality score
    risk_pct:  float        # current risk-to-stop, as % of equity (0 if at BE+)
    direction: int          # +1 long / -1 short
    pnl_r:     float = 0.0   # live P&L in R (negative = losing)
    ticket:    int = 0


@dataclass
class Trim:
    """Reduce an open position's risk by `reduce_pct` (% of equity)."""
    symbol:     str
    ticket:     int
    reduce_pct: float


@dataclass
class Allocation:
    """Result of sizing a new trade against the live book."""
    granted_pct: float            # risk % of equity to give the new trade (0 = skip)
    trims:       list = field(default_factory=list)   # list[Trim] to free budget
    reason:      str = ""

    @property
    def taken(self) -> bool:
        return self.granted_pct > 0


class PortfolioAllocator:
    """Active, quality-weighted risk allocation across concurrent trades.

    daily_budget_pct : max aggregate open risk (also the daily-loss ceiling we
                       actively defend), e.g. 4.0
    score_edge       : how much better (score) a new trade must be than an open
                       one before we trim the open one to fund it.
    min_trade_pct    : smallest risk worth taking; below this we skip rather than
                       open a token position.
    """

    def __init__(self, daily_budget_pct: float = 4.0, score_edge: int = 1,
                 min_trade_pct: float = 0.25):
        self.budget      = float(daily_budget_pct)
        self.score_edge  = int(score_edge)
        self.min_trade   = float(min_trade_pct)

    # ── Behaviour 1 + 2: size a new trade, reallocating if needed ─────────────

    def allocate(self, new_score: int, intended_pct: float,
                 open_positions: list[OpenPos]) -> Allocation:
        open_risk = sum(p.risk_pct for p in open_positions)
        room = self.budget - open_risk

        # Plenty of budget → grant the intended size (bounded by remaining room).
        if intended_pct <= room:
            return Allocation(round(intended_pct, 2), [], "within budget")

        # Some room, but not the full intended size. Grant what fits; if the new
        # setup is clearly stronger than the weakest open trade, trim that trade
        # to top the new one back up toward its intended size (behaviour 2).
        trims: list[Trim] = []
        granted = max(room, 0.0)
        need = intended_pct - granted

        if need > 0:
            # Candidates to trim: weaker (by score_edge) and/or losing trades,
            # weakest & most-losing first.
            candidates = sorted(
                (p for p in open_positions
                 if p.risk_pct > 0 and (new_score - p.score) >= self.score_edge),
                key=lambda p: (p.score, p.pnl_r),   # lowest score, then most losing
            )
            for p in candidates:
                if need <= 0:
                    break
                # Trim up to half a weaker position's risk to fund the better one.
                free = min(p.risk_pct * 0.5, need)
                if free <= 0:
                    continue
                trims.append(Trim(p.symbol, p.ticket, round(free, 2)))
                granted += free
                need    -= free

        if granted < self.min_trade:
            return Allocation(0.0, [], f"no budget (open {open_risk:.2f}%, "
                                       f"room {room:.2f}%, no weaker trade to trim)")

        why = "within budget" if not trims else \
              f"reallocated {sum(t.reduce_pct for t in trims):.2f}% from " \
              f"{len(trims)} weaker/losing trade(s)"
        return Allocation(round(min(granted, intended_pct), 2), trims, why)

    # ── Behaviour 3: keep aggregate risk under the daily ceiling ──────────────

    def daily_guard(self, day_loss_pct: float, open_positions: list[OpenPos],
                    limit_pct: float = 5.0, buffer_pct: float = 1.0) -> list[Trim]:
        """Return trims so that (day loss already taken + remaining open risk)
        cannot breach `limit_pct`. Keeps a safety `buffer_pct` below the hard
        limit. Trims the weakest/most-losing exposure first."""
        headroom = (limit_pct - buffer_pct) - day_loss_pct
        open_risk = sum(p.risk_pct for p in open_positions)
        if open_risk <= headroom:
            return []

        excess = open_risk - max(headroom, 0.0)
        trims: list[Trim] = []
        # Shed the weakest, most-losing risk first until within headroom.
        for p in sorted(open_positions, key=lambda p: (p.score, p.pnl_r)):
            if excess <= 0:
                break
            cut = min(p.risk_pct, excess)
            if cut <= 0:
                continue
            trims.append(Trim(p.symbol, p.ticket, round(cut, 2)))
            excess -= cut
        return trims
