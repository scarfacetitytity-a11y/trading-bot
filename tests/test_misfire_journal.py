"""Verify the journaling + misfire self-improvement loop end to end.

The knowledge base is built from this: every loss must land in the ledger with
its confluences (reasons) and the right diagnostic flags, and misfire_report()
must aggregate them. If this breaks, the bot stops learning.
"""
import json
from unittest import mock

import execution.trade_journal as tj


def _journal(tmp_path):
    # silence Telegram + git during the test
    with mock.patch.object(tj.tg, "notify_trade_open", lambda **k: None), \
         mock.patch.object(tj.tg, "notify_trade_close", lambda **k: None), \
         mock.patch.object(tj.tg, "notify_council_flag", lambda *a, **k: None):
        yield tj.TradeJournal(log_dir=str(tmp_path))


class TestMisfireLoop:
    def test_loss_lands_in_ledger_with_reasons_and_flags(self, tmp_path):
        with mock.patch.object(tj.tg, "notify_trade_open", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_trade_close", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_council_flag", lambda *a, **k: None):
            j = tj.TradeJournal(log_dir=str(tmp_path))

            # A SHORT that loses: entry 100, stop 100.5 (0.5 wide), closes at 100.5 (-1R)
            j.open_trade(symbol="US500.cash", direction=-1, score=4,
                         entry_price=100.0, sl_price=100.5, tp_price=97.5,
                         lots=2.0, equity=95_000, atr=2.0,
                         grade="C", trade_type="continuation",
                         reasons=["H4 bias +2", "Order block"])
            j.close_trade("US500.cash", close_price=100.5, equity_after=94_500)

            ledger = tmp_path / "misfires.jsonl"
            assert ledger.exists(), "misfire ledger not written"
            rec = json.loads(ledger.read_text().strip().splitlines()[-1])

            assert rec["symbol"] == "US500.cash"
            assert rec["score"] == 4
            assert rec["r_multiple"] < 0
            assert rec["reasons"] == ["H4 bias +2", "Order block"]
            # tight stop: 0.5 / atr 2.0 = 0.25 ATR < 0.6 -> flagged; grade C -> low_grade
            assert "tight_stop" in rec["flags"]
            assert "low_grade" in rec["flags"]

    def test_wins_do_not_land_in_ledger(self, tmp_path):
        with mock.patch.object(tj.tg, "notify_trade_open", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_trade_close", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_council_flag", lambda *a, **k: None):
            j = tj.TradeJournal(log_dir=str(tmp_path))
            j.open_trade(symbol="XAUUSD", direction=1, score=6,
                         entry_price=3300.0, sl_price=3295.0, tp_price=3320.0,
                         lots=1.0, equity=95_000, atr=8.0,
                         grade="A", trade_type="continuation",
                         reasons=["H4 bias +2", "Liquidity sweep", "RSI zone"])
            j.close_trade("XAUUSD", close_price=3320.0, equity_after=95_400)  # win
            assert not (tmp_path / "misfires.jsonl").exists()

    def test_report_aggregates_patterns(self, tmp_path):
        with mock.patch.object(tj.tg, "notify_trade_open", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_trade_close", lambda **k: None), \
             mock.patch.object(tj.tg, "notify_council_flag", lambda *a, **k: None):
            j = tj.TradeJournal(log_dir=str(tmp_path))
            for i in range(3):
                j.open_trade(symbol="US30.cash", direction=-1, score=4,
                             entry_price=44000.0, sl_price=44030.0, tp_price=43900.0,
                             lots=1.0, equity=95_000, atr=120.0,
                             grade="C", trade_type="continuation",
                             reasons=["H4 bias +2", "Order block"])
                j.close_trade("US30.cash", close_price=44030.0, equity_after=94_900)
            rep = j.misfire_report()
            assert rep["count"] == 3
            assert rep["avg_r"] < 0
            flags = dict(rep["top_flags"])
            assert flags.get("low_grade") == 3
            assert ("continuation", 3) in rep["losers_by_type"]
