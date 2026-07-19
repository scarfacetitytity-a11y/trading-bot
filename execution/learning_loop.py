"""Automatic learning loop — analyze every trade, PROPOSE changes, never apply.

Reads the durable trade record (MT5 deal history + the journal's context +
the misfire ledger), mines what keeps losing, and writes structured PROPOSALS
to logs/learning_proposals.md for Anton to review.

HARD RULE (Anton, 2026-07-19): this NEVER edits the strategy or config. It only
proposes. A human approves every change — the loop can be wrong about what
matters, so removal/tightening is always human-gated.

Usage:  venv/Scripts/python.exe -m execution.learning_loop
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
PROPOSALS = LOG_DIR / "learning_proposals.md"
MAGIC = 234001


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(name: str) -> list:
    p = LOG_DIR / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _deal_history(days: int = 30) -> list:
    """Durable closed-trade truth from MT5, grouped per position_id."""
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []
    if not mt5.initialize():
        return []
    now = datetime.now()
    deals = mt5.history_deals_get(now - timedelta(days=days), now + timedelta(minutes=1)) or []
    by_pos = defaultdict(list)
    for d in deals:
        if d.magic == MAGIC:
            by_pos[d.position_id].append(d)
    trades = []
    for pid, dl in by_pos.items():
        outs = [d for d in dl if d.entry in (1, 3)]
        ins  = [d for d in dl if d.entry == 0]
        if not outs or not ins:
            continue
        trades.append({
            "pos": pid, "symbol": dl[0].symbol,
            "direction": -1 if ins[0].type == 1 else 1,
            "lots": round(sum(d.volume for d in ins), 2),
            "pnl": round(sum(d.profit + d.swap + d.commission for d in dl), 2),
        })
    mt5.shutdown()
    return trades


# ── Analysis → proposals (evidence-based, conservative) ───────────────────────

def analyze(days: int = 30) -> dict:
    deals    = _deal_history(days)
    misfires = _load_jsonl("misfires.jsonl")
    wins     = _load_jsonl("wins.jsonl")
    proposals = []

    def propose(title, evidence, change, confidence):
        proposals.append(dict(title=title, evidence=evidence, change=change,
                              confidence=confidence, status="NEEDS APPROVAL"))

    # ── Per-symbol performance (durable pnl) ──
    by_sym = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in deals:
        s = by_sym[t["symbol"]]
        s["n"] += 1; s["pnl"] += t["pnl"]; s["wins"] += (t["pnl"] > 0)
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
    for t in deals:
        d = dir_sym[(t["symbol"], t["direction"])]
        d["n"] += 1; d["pnl"] += t["pnl"]
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
    if deals:
        lots = sorted(t["lots"] for t in deals)
        med = lots[len(lots) // 2]
        big = [t for t in deals if t["lots"] > med * 10 and t["lots"] > 20]
        if big:
            worst = min(big, key=lambda t: t["pnl"])
            propose(
                f"Runaway lot sizes ({len(big)} trades > 10x median {med})",
                f"e.g. {worst['symbol']} {worst['lots']} lots -> ${worst['pnl']:+.0f}",
                "notional cap + min-stop guards now added — verify no new trade exceeds them.",
                "high")

    return dict(deals=len(deals), misfires=len(misfires), wins=len(wins), proposals=proposals)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(result: dict) -> None:
    lines = [
        "---", "type: analysis", "status: needs-review",
        "tags: [trading-bot, learning-loop, proposals, risk]",
        "relatedTo: [trading-bot, misfire-ledger]", "---", "",
        f"# Learning-loop proposals — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
        f"Analyzed **{result['deals']}** closed trades, **{result.get('wins',0)}** wins, "
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
    print(f"Learning loop: {res['deals']} trades, {res['misfires']} misfires "
          f"-> {len(res['proposals'])} PROPOSALS (needs approval)")
    for p in res["proposals"]:
        print(f"  [{p['confidence']:>6}] {p['title']}")
    print(f"\nFull report (review + approve): {PROPOSALS}")


if __name__ == "__main__":
    main()
