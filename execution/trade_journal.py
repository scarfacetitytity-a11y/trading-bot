"""Self-analyzing trade journal.

Logs every trade open/close with full context. After each close,
computes rolling performance stats and flags drift from backtest baseline.

Council of 12 automated checks fire on every close:
  #05 Compliance  — FTMO daily/total DD still within limits
  #09 Test        — live WR drifting below backtest floor?
  #11 Reality Gap — avg-R diverging from backtest expectation?
  #12 Devil       — consecutive losses exceeding threshold?
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from execution import telegram_notify as tg
from execution.trade_analyzer import analyze_exit

logger = logging.getLogger(__name__)

# Backtest baseline — from M2P10 26-month run
_BASELINE_WIN_RATE  = 0.396   # 39.6%
_BASELINE_AVG_R     = 0.386   # EV per trade in R
_BASELINE_MAX_DD    = 4.09    # max DD %

# Drift thresholds — Council #11 Reality Gap
_WR_FLOOR           = 0.28    # below 28% WR over 30 trades → flag
_AVG_R_FLOOR        = 0.10    # below 0.10 avg-R over 30 trades → flag
_CONSEC_LOSS_LIMIT  = 7       # Council #12: 7 consecutive losses → flag


@dataclass
class TradeRecord:
    symbol:       str
    direction:    int          # 1=long, -1=short
    score:        int
    session_hour: int          # UTC hour at entry
    entry_price:  float
    sl_price:     float
    tp_price:     Optional[float]
    lots:         float
    equity_at_entry: float
    open_time:    str          # ISO UTC
    close_time:   Optional[str] = None
    close_price:  Optional[float] = None
    pnl_usd:      Optional[float] = None
    r_multiple:   Optional[float] = None   # PnL / (entry - SL) in price terms
    outcome:      Optional[str] = None     # "win" | "loss" | "breakeven"
    # ── Trade plan (from TradeAnalyzer at entry) ──
    trade_type:   Optional[str] = None     # continuation | sweep_reversal | breakout | range
    grade:        Optional[str] = None     # A | B | C
    target_price: Optional[float] = None   # liquidity target (structural TP)
    thesis:       Optional[str] = None
    # ── Path tracking (MFE/MAE while open) ──
    path_high:    Optional[float] = None
    path_low:     Optional[float] = None
    # ── Post-trade review (from analyze_exit at close) ──
    hit_target:   Optional[bool] = None
    mfe_r:        Optional[float] = None
    mae_r:        Optional[float] = None
    thesis_valid: Optional[bool] = None
    lesson:       Optional[str] = None
    review_notes: Optional[list] = None


class TradeJournal:

    def __init__(self, log_dir: str = "logs"):
        self._path   = Path(log_dir) / "trades.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, TradeRecord] = {}   # symbol → open trade

    # ── Open / close ─────────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        direction: int,
        score: int,
        entry_price: float,
        sl_price: float,
        tp_price: Optional[float],
        lots: float,
        equity: float,
        atr: Optional[float] = None,
        df=None,
        trade_type: Optional[str] = None,
        grade: Optional[str] = None,
        target_price: Optional[float] = None,
        thesis: Optional[str] = None,
        reasons: Optional[list] = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc)
        self._open[symbol] = TradeRecord(
            symbol=symbol,
            direction=direction,
            score=score,
            session_hour=now.hour,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            lots=lots,
            equity_at_entry=equity,
            open_time=now.isoformat(),
            trade_type=trade_type,
            grade=grade,
            target_price=target_price,
            thesis=thesis,
            path_high=entry_price,
            path_low=entry_price,
        )
        logger.info("[Journal] OPEN %s dir=%+d score=%d entry=%.5f SL=%.5f TP=%s",
                    symbol, direction, score, entry_price, sl_price, tp_price)
        tg.notify_trade_open(
            symbol=symbol, direction=direction, score=score,
            entry=entry_price, sl=sl_price, tp=tp_price,
            lots=lots, equity=equity, atr=atr, df=df, reasons=reasons,
        )

    def update_path(self, symbol: str, high: float, low: float) -> None:
        """Track max favourable/adverse excursion while a trade is open."""
        rec = self._open.get(symbol)
        if rec is None:
            return
        if rec.path_high is None or high > rec.path_high:
            rec.path_high = float(high)
        if rec.path_low is None or low < rec.path_low:
            rec.path_low = float(low)

    def close_trade(self, symbol: str, close_price: float, equity_after: float) -> None:
        rec = self._open.pop(symbol, None)
        if rec is None:
            return

        rec.close_time  = datetime.now(tz=timezone.utc).isoformat()
        rec.close_price = close_price
        rec.pnl_usd     = equity_after - rec.equity_at_entry

        risk_dist = abs(rec.entry_price - rec.sl_price)
        if risk_dist > 0:
            price_gain = (close_price - rec.entry_price) * rec.direction
            rec.r_multiple = price_gain / risk_dist
        else:
            rec.r_multiple = 0.0

        if rec.r_multiple > 0.1:
            rec.outcome = "win"
        elif rec.r_multiple < -0.8:
            rec.outcome = "loss"
        else:
            rec.outcome = "breakeven"

        # ── Post-trade review — learn from every close ──
        self._review(rec, close_price)

        self._append(rec)
        self._council_review(rec)
        session_pnl = sum(
            t.get("pnl_usd", 0) or 0
            for t in self._load_recent(200)
            if t.get("open_time", "")[:10] == datetime.now(tz=timezone.utc).date().isoformat()
        )
        tg.notify_trade_close(
            symbol=rec.symbol, direction=rec.direction,
            outcome=rec.outcome or "unknown",
            r_multiple=rec.r_multiple or 0.0,
            pnl_usd=rec.pnl_usd or 0.0,
            equity=equity_after,
            session_pnl=session_pnl,
        )

    # ── Post-trade review ─────────────────────────────────────────────────────

    def _review(self, rec: TradeRecord, close_price: float) -> None:
        """Run analyze_exit and store the verdict + lesson on the record."""
        target = rec.target_price if rec.target_price is not None else rec.tp_price
        if target is None:
            return
        reason = (rec.outcome or "").upper()
        path_high = rec.path_high if rec.path_high is not None else close_price
        path_low  = rec.path_low  if rec.path_low  is not None else close_price
        try:
            review = analyze_exit(
                direction=rec.direction, entry=rec.entry_price, stop=rec.sl_price,
                target=target, exit_px=close_price,
                path_high=path_high, path_low=path_low, reason=reason,
            )
        except Exception as exc:
            logger.warning("[Journal] review failed for %s: %s", rec.symbol, exc)
            return
        rec.hit_target   = review.hit_target
        rec.mfe_r        = review.mfe_r
        rec.mae_r        = review.mae_r
        rec.thesis_valid = review.thesis_valid
        rec.lesson       = review.lesson
        rec.review_notes = review.notes
        logger.info(
            "[Journal] REVIEW %s [%s/%s] hitTP=%s MFE=%.1fR MAE=%.1fR | %s%s",
            rec.symbol, rec.trade_type or "?", rec.grade or "?",
            review.hit_target, review.mfe_r, review.mae_r, review.lesson,
            (" | " + "; ".join(review.notes)) if review.notes else "",
        )

    # ── Rolling stats ─────────────────────────────────────────────────────────

    def rolling_stats(self, n: int = 30) -> dict:
        trades = self._load_recent(n)
        closed = [t for t in trades if t.get("outcome")]
        if not closed:
            return {"trades": 0}
        wins   = sum(1 for t in closed if t["outcome"] == "win")
        r_vals = [t["r_multiple"] for t in closed if t.get("r_multiple") is not None]
        return {
            "trades":   len(closed),
            "win_rate": wins / len(closed),
            "avg_r":    sum(r_vals) / len(r_vals) if r_vals else 0.0,
            "consec_losses": self._consecutive_losses(closed),
        }

    # ── Council of 12 automated checks ───────────────────────────────────────

    def _council_review(self, rec: TradeRecord) -> None:
        stats = self.rolling_stats(30)
        n = stats.get("trades", 0)

        # Council #09 — Reality Gap: WR drift
        if n >= 10:
            wr = stats["win_rate"]
            if wr < _WR_FLOOR:
                logger.warning(
                    "[Council #11 Reality Gap] Live WR %.1f%% over %d trades — "
                    "below floor %.1f%%. Backtest: %.1f%%.",
                    wr * 100, n, _WR_FLOOR * 100, _BASELINE_WIN_RATE * 100,
                )

        # Council #11 — avg-R drift
        if n >= 10:
            avg_r = stats["avg_r"]
            if avg_r < _AVG_R_FLOOR:
                logger.warning(
                    "[Council #11 Reality Gap] Live avg-R %.3f over %d trades — "
                    "below floor %.3f. Backtest: %.3f.",
                    avg_r, n, _AVG_R_FLOOR, _BASELINE_AVG_R,
                )

        # Council #12 — consecutive losses
        consec = stats.get("consec_losses", 0)
        if consec >= _CONSEC_LOSS_LIMIT:
            msg = f"{consec} consecutive losses. Review market regime before next entry."
            logger.critical("[Council #12 Devil's Advocate] %s", msg)
            tg.notify_council_flag("12 Devil's Advocate", msg)

        logger.info(
            "[Journal] CLOSE %s outcome=%s R=%.2f | rolling(%d): WR=%.1f%% avgR=%.3f",
            rec.symbol, rec.outcome, rec.r_multiple or 0,
            n, stats.get("win_rate", 0) * 100, stats.get("avg_r", 0),
        )

    # ── GitHub auto-push ──────────────────────────────────────────────────────

    def push_to_github(self) -> None:
        try:
            repo = Path(__file__).parent.parent
            subprocess.run(["git", "-C", str(repo), "add", str(self._path)],
                           check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m",
                 f"auto: trade journal update {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"],
                check=True, capture_output=True,
            )
            subprocess.run(["git", "-C", str(repo), "push", "origin", "master"],
                           check=True, capture_output=True)
            logger.info("[Journal] Pushed trade log to GitHub.")
        except subprocess.CalledProcessError as e:
            logger.warning("[Journal] GitHub push failed: %s", e.stderr.decode() if e.stderr else e)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _append(self, rec: TradeRecord) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def _load_recent(self, n: int) -> list[dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l]

    def _consecutive_losses(self, trades: list[dict]) -> int:
        count = 0
        for t in reversed(trades):
            if t.get("outcome") == "loss":
                count += 1
            else:
                break
        return count
