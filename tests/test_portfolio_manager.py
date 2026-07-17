"""Unit tests for the adaptive PortfolioAllocator (behaviours 1-3).

Pure logic — no broker connection. Verifies quality-weighted sizing,
reallocation from weaker/losing trades, and the active daily-loss guarantee.
"""
from execution.portfolio_manager import PortfolioAllocator, OpenPos, Trim


def _alloc():
    return PortfolioAllocator(daily_budget_pct=4.0, score_edge=1, min_trade_pct=0.25)


# ── Behaviour 1: adapt to remaining budget, never token-size ──────────────────

class TestAdaptiveSizing:
    def test_full_size_when_budget_free(self):
        a = _alloc().allocate(6, intended_pct=3.0, open_positions=[])
        assert a.taken and a.granted_pct == 3.0 and a.trims == []

    def test_scaled_to_remaining_room(self):
        # 3% already open, 4% budget → only 1% room, no weaker trade to trim.
        book = [OpenPos("XAUUSD", score=6, risk_pct=3.0, direction=1, pnl_r=0.5)]
        a = _alloc().allocate(6, intended_pct=2.0, open_positions=book)
        assert a.granted_pct == 1.0 and a.trims == []

    def test_skipped_only_when_no_budget_and_no_weaker_trade(self):
        # Budget full, open trade is equal/higher quality → cannot trim → skip.
        book = [OpenPos("XAUUSD", score=6, risk_pct=4.0, direction=1, pnl_r=1.0)]
        a = _alloc().allocate(5, intended_pct=2.0, open_positions=book)
        assert not a.taken and a.granted_pct == 0.0


# ── Behaviour 2: quality-weighted reallocation ────────────────────────────────

class TestReallocation:
    def test_trims_weaker_losing_trade_to_fund_better(self):
        # Budget full with a weak, losing trade; a stronger setup arrives.
        book = [OpenPos("US100.cash", score=4, risk_pct=4.0, direction=1, pnl_r=-0.8, ticket=11)]
        a = _alloc().allocate(6, intended_pct=2.0, open_positions=book)
        assert a.taken
        assert a.trims and a.trims[0].symbol == "US100.cash"
        # trims up to half the weaker trade's risk (2.0%), funding the new one
        assert a.trims[0].reduce_pct == 2.0
        assert a.granted_pct == 2.0
        assert "reallocated" in a.reason

    def test_does_not_trim_stronger_or_equal_trades(self):
        book = [OpenPos("XAUUSD", score=7, risk_pct=4.0, direction=1, pnl_r=-0.5, ticket=9)]
        a = _alloc().allocate(6, intended_pct=2.0, open_positions=book)
        assert not a.taken            # new trade is weaker → no trim, no room
        assert a.trims == []

    def test_prefers_weakest_most_losing_first(self):
        book = [
            OpenPos("A", score=5, risk_pct=2.0, direction=1, pnl_r=-0.2, ticket=1),
            OpenPos("B", score=4, risk_pct=2.0, direction=1, pnl_r=-0.9, ticket=2),
        ]
        a = _alloc().allocate(6, intended_pct=2.0, open_positions=book)
        # room is 0; trims should start with the weakest/most-losing (B, score 4)
        assert a.trims[0].symbol == "B"


# ── Behaviour 3: active daily-loss guarantee ──────────────────────────────────

class TestDailyGuard:
    def test_no_trim_when_within_headroom(self):
        book = [OpenPos("XAUUSD", score=6, risk_pct=2.0, direction=1)]
        trims = _alloc().daily_guard(day_loss_pct=0.5, open_positions=book,
                                     limit_pct=5.0, buffer_pct=1.0)
        assert trims == []

    def test_trims_excess_to_stay_under_limit(self):
        # 3% already lost today, 3% still open, limit 5% with 1% buffer →
        # headroom = (5-1)-3 = 1% → must shed 2% of the 3% open.
        book = [
            OpenPos("A", score=6, risk_pct=1.5, direction=1, pnl_r=0.3, ticket=1),
            OpenPos("B", score=4, risk_pct=1.5, direction=1, pnl_r=-0.7, ticket=2),
        ]
        trims = _alloc().daily_guard(day_loss_pct=3.0, open_positions=book)
        assert round(sum(t.reduce_pct for t in trims), 2) == 2.0
        assert trims[0].symbol == "B"    # weakest/most-losing shed first
