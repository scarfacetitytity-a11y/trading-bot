"""Automatic learning loop — analyze live forward-test trades, PROPOSE changes, never apply.

Reads only the live journal files (trades.jsonl, misfires.jsonl, wins.jsonl) —
the durable record of every real trade the bot has taken. No backtesting data.

HARD RULE (Anton, 2026-07-19): this NEVER edits the strategy or config. It only
proposes. A human approves every change — the loop can be wrong about what
matters, so removal/tightening is always human-gated.

Forward testing only — keep backtesting analysis separate (Anton, 2026-07-20).

Usage:  venv/Scripts/python.exe -m execution.learning_loop
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
PROPOSALS = LOG_DIR / "learning_proposals.md"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(name: str) -> list:
    p = LOG_DIR / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _live_trades() -> list:
    """All closed trades from the live journal — forward-test data only."""
    trades = _load_jsonl("trades.jsonl")
    return [t for t in trades if t.get("outcome")]   # closed only


# ── Analysis → proposals (evidence-based, conservative) ───────────────────────

def analyze(days: int = 0) -> dict:  # days param kept for API compat, ignored
    trades   = _live_trades()
    misfires = _load_jsonl("misfires.jsonl")
    wins     = _load_jsonl("wins.jsonl")
    proposals = []

    def propose(title, evidence, change, confidence):
        proposals.append(dict(title=title, evidence=evidence, change=change,
                              confidence=confidence, status="NEEDS APPROVAL"))

    # ── Per-symbol performance (live journal pnl) ──
    by_sym = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        s = by_sym[t["symbol"]]
        s["n"] += 1
        s["pnl"] += t.get("pnl_usd", 0) or 0
        s["wins"] += (t.get("outcome") == "win")
    for sym, s in by_sym.items():
        if s["n"] >= 5:
            wr = s["wins"] / s["n"] * 100
            if wr < 30 and s["pnl"] < 0:
                propose(
                    f"{sym}: chronically negative ({wr:.0f}% WR, ${s['pnl']:+.0f} over {s['n']})",
                    f"{s['n']} trades, {s['wins']} wins, net ${s['pnl']:+.0f}",
                    f"Raise min_entry_score for {sym}, or restrict to its winning direction "
                    f"— DO NOT auto-exclude; confirm the setups aren't just pre-fix.",
                    "medium")
            elif wr >= 50 and s["pnl"] > 0:
                propose(
                    f"{sym}: positive edge ({wr:.0f}% WR, ${s['pnl']:+.0f}) — protect it",
                    f"{s['n']} trades, net ${s['pnl']:+.0f}",
                    f"Consider size-up / priority for {sym}; do not tighten its gates.",
                    "medium")

    # ── Direction bias per symbol ──
    dir_sym = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        d = dir_sym[(t["symbol"], t.get("direction", 0))]
        d["n"] += 1; d["pnl"] += t.get("pnl_usd", 0) or 0
    for (sym, direction), d in dir_sym.items():
        if d["n"] >= 4 and d["pnl"] < -200:
            side = "SHORT" if direction == -1 else "LONG"
            propose(
                f"{sym} {side}s bleeding (${d['pnl']:+.0f} over {d['n']})",
                f"{d['n']} {side} trades net ${d['pnl']:+.0f}",
                f"Raise the counter-trend score bar for {sym} {side}s (min_entry_score_counter), "
                f"or require an extra confluence for this side. Keep bidirectional.",
                "medium")

    # ── Misfire flag frequencies ──
    if misfires:
        fl = Counter(f for m in misfires for f in m.get("flags", []))
        n = len(misfires)
        for flag, ct in fl.most_common():
            pct = ct / n * 100
            if pct < 25:
                continue
            change = {
                "tight_stop":   "min_stop_atr_mult / min_stop_pct guards should cut these — verify post-fix count drops.",
                "low_grade":    "min_entry_score is now gating grade — verify grade-C count drops post-fix.",
                "never_worked": "ENTRY LOCATION problem — trades never went green. Needs an entry-timing rule (mentor pipeline).",
                "gave_back":    "MANAGEMENT problem — was >1.5R then closed red. Tighten trail / bank-in TP sooner.",
            }.get(flag, "investigate")
            propose(f"'{flag}' in {pct:.0f}% of losers", f"{ct}/{n} misfires flagged {flag}", change,
                    "high" if pct > 50 else "medium")

    # ── What WORKS: confluences over-represented in winners vs losers ──
    if wins:
        win_conf  = Counter(x for m in wins for x in m.get("reasons", []))
        loss_conf = Counter(x for m in misfires for x in m.get("reasons", []))
        nw, nl = len(wins), max(1, len(misfires))
        for conf, wc in win_conf.most_common():
            win_rate  = wc / nw
            loss_rate = loss_conf.get(conf, 0) / nl
            if win_rate >= 0.5 and win_rate > loss_rate * 1.3:
                propose(
                    f"Confluence '{conf}' is a WINNER (in {wc}/{nw} wins)",
                    f"appears in {win_rate*100:.0f}% of wins vs {loss_rate*100:.0f}% of losses",
                    f"Protect/amplify: weight '{conf}' higher in the score, or prioritise setups with it.",
                    "medium")
        # best-performing trade type
        wtype = Counter(m.get("trade_type") for m in wins).most_common(1)
        if wtype and wtype[0][1] >= 3:
            propose(
                f"'{wtype[0][0]}' setups win most ({wtype[0][1]} wins)",
                f"{wtype[0][1]} winning {wtype[0][0]} trades",
                f"Prioritise / size-up '{wtype[0][0]}' setups.", "low")

    # ── Size anomalies (runaway lots) ──
    if trades:
        all_lots = sorted(t.get("lots", 0) for t in trades if t.get("lots"))
        if all_lots:
            med = all_lots[len(all_lots) // 2]
            big = [t for t in trades if (t.get("lots") or 0) > med * 10 and (t.get("lots") or 0) > 20]
            if big:
                worst = min(big, key=lambda t: t.get("pnl_usd", 0) or 0)
                propose(
                    f"Runaway lot sizes ({len(big)} trades > 10x median {med})",
                    f"e.g. {worst['symbol']} {worst.get('lots')} lots -> ${worst.get('pnl_usd',0):+.0f}",
                    "notional cap + min-stop guards now added — verify no new trade exceeds them.",
                    "high")

    return dict(trades=len(trades), misfires=len(misfires), wins=len(wins), proposals=proposals)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(result: dict) -> None:
    lines = [
        "---", "type: analysis", "status: needs-review",
        "tags: [trading-bot, learning-loop, proposals, risk]",
        "relatedTo: [trading-bot, misfire-ledger]", "---", "",
        f"# Learning-loop proposals — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
        f"Analyzed **{result['trades']}** live trades, **{result.get('wins',0)}** wins, "
        f"**{result['misfires']}** misfires. "
        f"**{len(result['proposals'])} proposals — NONE applied. Every one needs your approval.**", "",
    ]
    if not result["proposals"]:
        lines.append("_No actionable patterns yet (need more post-fix trades)._")
    for i, p in enumerate(result["proposals"], 1):
        lines += [
            f"## {i}. {p['title']}  `[{p['status']}]`",
            f"- **Evidence:** {p['evidence']}",
            f"- **Proposed change:** {p['change']}",
            f"- **Confidence:** {p['confidence']}", "",
        ]
    PROPOSALS.write_text("\n".join(lines), encoding="utf-8")


def main():
    res = analyze()
    write_report(res)
    print(f"Learning loop: {res['trades']} live trades, {res['misfires']} misfires "
          f"-> {len(res['proposals'])} PROPOSALS (needs approval)")
    for p in res["proposals"]:
        print(f"  [{p['confidence']:>6}] {p['title']}")
    print(f"\nFull report (review + approve): {PROPOSALS}")


if __name__ == "__main__":
    main()
