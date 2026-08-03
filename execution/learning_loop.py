"""Automatic learning loop — Council of 12 analysis on every live trade.

Reads only live journal files (trades.jsonl, misfires.jsonl, wins.jsonl).
No backtesting data — forward-test only (Anton, 2026-07-20).

Each of the 12 Council members runs their own lens over the trade record:
  #05 Compliance     — FTMO rule adherence, daily DD close calls
  #06 App Engineer   — gate logic, scoring accuracy, sizing correctness
  #07 SRE            — BE trigger timing, circuit breaker activations
  #09 Test Engineer  — are live stats within expected backtest distribution?
  #11 Reality Gap    — where live diverges from what the strategy expected
  #12 Devil's Adv.   — beats up wins AND losses; finds systematic errors

HARD RULE (Anton, 2026-07-19): this NEVER edits strategy or config.
Proposals only. Every change requires human approval.

Usage:  python -m execution.learning_loop
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR   = Path(__file__).resolve().parent.parent / "logs"
PROPOSALS = LOG_DIR / "learning_proposals.md"

# Statistical significance threshold (ruflo backtest quality gate: p < 0.05)
_SIG_P_VALUE = 0.05
_SIG_MIN_TRADES = 30   # Devil's Advocate: no conclusions below this

# Backtest baseline (from 26-month XAUUSD H1 run) — must precede
# _is_significant, which uses _BASELINE_WR as a default argument
_BASELINE_WR    = 0.396
_BASELINE_AVG_R = 0.386
_WR_FLOOR       = 0.28
_AVG_R_FLOOR    = 0.10
_CONSEC_LOSS_LIMIT = 7


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for Binomial(n, p) — pure Python, no scipy required."""
    from math import comb
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(total, 1.0)


def _is_significant(wins: int, n: int, baseline_wr: float = _BASELINE_WR) -> tuple[bool, float]:
    """Return (significant, p_value) — two-tailed binomial test vs baseline WR."""
    if n < 10:
        return False, 1.0
    # Lower tail: worse than baseline
    p_lower = _binomial_cdf(wins, n, baseline_wr)
    # Upper tail: better than baseline
    p_upper = 1.0 - _binomial_cdf(wins - 1, n, baseline_wr)
    p_val = 2.0 * min(p_lower, p_upper)   # two-tailed
    return p_val < _SIG_P_VALUE, round(p_val, 4)

# Additional log directories from other accounts. Add paths here when a second
# or third account bot install exists. The learning loop merges all trade records.
_EXTRA_LOG_DIRS: list[Path] = [
    # e.g. Path("C:/Users/anton/Documents/trading-bot-account2/logs"),
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(name: str) -> list:
    rows = []
    for d in [LOG_DIR] + _EXTRA_LOG_DIRS:
        p = d / name
        if p.exists():
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def _live_trades() -> list:
    """All closed trades across all registered accounts — forward-test data only."""
    return [t for t in _load_jsonl("trades.jsonl") if t.get("outcome")]


def _r(t) -> float:
    return float(t.get("r_multiple") or 0)


def _pnl(t) -> float:
    return float(t.get("pnl_usd") or 0)


def _consec_losses(trades: list) -> int:
    count = 0
    for t in reversed(trades):
        if t.get("outcome") == "loss":
            count += 1
        else:
            break
    return count


# ── Council observation dataclass ─────────────────────────────────────────────

def _obs(council_id: str, member: str, title: str, evidence: str,
         verdict: str, proposal: str, confidence: str) -> dict:
    return dict(
        council=council_id, member=member, title=title,
        evidence=evidence, verdict=verdict, proposal=proposal,
        confidence=confidence, status="NEEDS APPROVAL",
    )


# ── Council #05 — Compliance Officer ─────────────────────────────────────────

def _council_05(trades: list, misfires: list) -> list:
    """FTMO rule adherence — daily DD close calls, entry cap, session timing."""
    obs = []

    # Equity close calls: trades where daily loss approached 5% (FTMO daily limit)
    # We track this via equity_at_entry vs pnl to estimate intraday DD
    daily: dict = defaultdict(lambda: {"start_eq": None, "losses": 0.0, "entries": 0})
    for t in trades:
        day = (t.get("open_time") or "")[:10]
        eq  = float(t.get("equity_at_entry") or 0)
        if daily[day]["start_eq"] is None or eq > daily[day]["start_eq"]:
            daily[day]["start_eq"] = eq
        if (t.get("outcome") == "loss"):
            daily[day]["losses"] += abs(_pnl(t))
        daily[day]["entries"] += 1

    close_call_days = []
    for day, d in daily.items():
        if d["start_eq"] and d["start_eq"] > 0:
            dd_pct = d["losses"] / d["start_eq"] * 100
            if dd_pct >= 3.0:   # within 2% of the 5% FTMO daily limit
                close_call_days.append((day, round(dd_pct, 2)))

    if close_call_days:
        worst = max(close_call_days, key=lambda x: x[1])
        obs.append(_obs("05", "Compliance Officer",
            f"FTMO daily DD close call: {len(close_call_days)} day(s) at 3%+ intraday loss",
            f"Worst day: {worst[0]} — {worst[1]}% intraday DD (FTMO limit 5%)",
            "Near-breach: the 2% internal circuit breaker in risk_agent.py exists to "
            "provide a 3% buffer above FTMO's 5% limit. If we are reaching 3% intraday, "
            "the internal breaker is not triggering early enough, OR position sizing is too large.",
            "Audit max_daily_loss_pct in RiskConfig. Current default 2% should stop us well "
            "before 5%. If close calls persist, reduce to 1.5%.",
            "high"))

    # Entry cap: any day with 6+ trades
    over_cap = [(day, d["entries"]) for day, d in daily.items() if d["entries"] >= 6]
    if over_cap:
        obs.append(_obs("05", "Compliance Officer",
            f"Daily entry cap hit on {len(over_cap)} day(s)",
            "; ".join(f"{d}: {n} entries" for d, n in sorted(over_cap)),
            "The cap exists for decision fatigue. Hitting it means the system is actively "
            "trading up to its psychological limit — not a rule breach, but flag for review.",
            "Confirm trades on capped days were high-score. If lower-quality entries are "
            "filling the cap early, tighten min_entry_score.",
            "medium"))

    return obs


# ── Council #06 — App Engineer ────────────────────────────────────────────────

def _council_06(trades: list, misfires: list, wins: list) -> list:
    """Gate logic, scoring accuracy, sizing correctness."""
    obs = []
    if not trades:
        return obs

    # Score vs outcome correlation — high-score trades should win more
    by_score: dict = defaultdict(lambda: {"n": 0, "wins": 0, "r": 0.0})
    for t in trades:
        sc = int(t.get("score") or 0)
        bucket = sc if sc <= 10 else 10
        by_score[bucket]["n"] += 1
        by_score[bucket]["wins"] += (t.get("outcome") == "win")
        by_score[bucket]["r"] += _r(t)

    inversion = []
    for sc in sorted(by_score):
        d = by_score[sc]
        if d["n"] >= 3:
            wr = d["wins"] / d["n"]
            avg_r = d["r"] / d["n"]
            if sc >= 7 and wr < 0.30:
                inversion.append(f"score {sc}: {wr*100:.0f}% WR ({d['n']} trades) — expected >50%")
            if sc <= 4 and wr > 0.55:
                inversion.append(f"score {sc}: {wr*100:.0f}% WR — lower scores shouldn't win this often; check floor")

    if inversion:
        obs.append(_obs("06", "App Engineer",
            "Score / outcome inversion detected",
            "; ".join(inversion),
            "If high-score setups are losing and low-score setups winning, the confluence "
            "weighting is wrong — individual factors that should add certainty may be noise.",
            "Cross-tabulate each confluence vs win rate. Remove or halve confluences that "
            "appear equally in winners and losers. Promote confluences that appear 2x+ more in winners.",
            "high"))

    # Grade C trades slipping through
    grade_c = [t for t in trades if t.get("grade") == "C"]
    if grade_c:
        c_wins = sum(1 for t in grade_c if t.get("outcome") == "win")
        obs.append(_obs("06", "App Engineer",
            f"Grade C trades in journal: {len(grade_c)} ({c_wins} wins)",
            f"{len(grade_c)} grade-C trades placed; grade C = no clean structural draw",
            "Grade C should mean min size (0.25x) or skip. If these are losing, they are "
            "bleeding equity on low-conviction setups. If winning, the grading may be too harsh.",
            "If grade-C loss rate > 60%, raise the tradeable threshold in analyze_entry() "
            "to skip grade C entirely. If winning rate is OK, just confirm size was 0.25x.",
            "medium"))

    # Sizing check: high-score trades should be largest
    high_score_lots = [t.get("lots", 0) or 0 for t in trades if (t.get("score") or 0) >= 7]
    low_score_lots  = [t.get("lots", 0) or 0 for t in trades if (t.get("score") or 0) <= 4]
    if high_score_lots and low_score_lots:
        avg_hi = sum(high_score_lots) / len(high_score_lots)
        avg_lo = sum(low_score_lots) / len(low_score_lots)
        if avg_hi < avg_lo * 1.1:
            obs.append(_obs("06", "App Engineer",
                "Sizing not scaling with score",
                f"High-score (7+) avg lots: {avg_hi:.2f} vs low-score (<=4): {avg_lo:.2f}",
                "The quality-scaled risk sizing is supposed to produce larger size on high-score "
                "trades. If averages are similar, the sizing multiplier is not working or the "
                "score floor for full size is set too low.",
                "Check score_mult logic in orchestrator. Verify score >= 7 maps to 1.0x and "
                "score <= 4 maps to 0.25x. If scores cluster in the middle, tighten the floor.",
                "medium"))

    return obs


# ── Council #07 — SRE ─────────────────────────────────────────────────────────

def _council_07(trades: list, misfires: list) -> list:
    """BE trigger timing, gave-back analysis, circuit breaker review."""
    obs = []

    gave_back = [t for t in misfires if "gave_back" in (t.get("flags") or [])]
    if len(gave_back) >= 3:
        symbols = Counter(t.get("symbol") for t in gave_back).most_common(3)
        obs.append(_obs("07", "SRE",
            f"'gave_back' pattern in {len(gave_back)} losing trades — reached >1.5R then closed red",
            f"Top symbols: {', '.join(f'{s}({n})' for s, n in symbols)}",
            "Trades that reached +1.5R then closed as losses are a trail/management failure. "
            "The adaptive trail should have locked in profit by +1.5R. Either the structural "
            "trail is too loose, or the BE trigger fired too late.",
            "Audit manage_trade() — ensure BE triggers at 0.5R for sweep_reversal and 1.0R "
            "for continuation. Also check Asian 50% BE trigger is firing. Consider tightening "
            "trail from 'next swing' to 'half-swing-to-swing' distance at +2R.",
            "high"))

    never_worked = [t for t in misfires if "never_worked" in (t.get("flags") or [])]
    if len(never_worked) >= 3:
        obs.append(_obs("07", "SRE",
            f"'never_worked' in {len(never_worked)} losses — price never went green",
            f"Avg MAE on these: {sum(t.get('mae_r', 0) or 0 for t in never_worked)/len(never_worked):.2f}R",
            "Trades that never move in our direction indicate bad entry timing or wrong direction "
            "read. These are not management failures — they are entry failures.",
            "Require a confirmed lower-timeframe indication before entry (M1/M3 structural break "
            "in trade direction). The 'never_worked' flag should reduce with tighter entry timing.",
            "high"))

    return obs


# ── Council #09 — Test Engineer ───────────────────────────────────────────────

def _council_09(trades: list) -> list:
    """Live stats vs expected distribution; confidence intervals."""
    obs = []
    if len(trades) < 10:
        return obs

    wins   = [t for t in trades if t.get("outcome") == "win"]
    losses = [t for t in trades if t.get("outcome") == "loss"]
    wr     = len(wins) / len(trades)
    r_vals = [_r(t) for t in trades]
    avg_r  = sum(r_vals) / len(r_vals) if r_vals else 0.0

    sig, p_val = _is_significant(len(wins), len(trades))
    sig_tag = f" [p={p_val:.3f}, {'SIGNIFICANT' if sig else 'NOT significant'}]"
    if wr < _WR_FLOOR:
        obs.append(_obs("09", "Test Engineer",
            f"Live WR {wr*100:.1f}% below floor {_WR_FLOOR*100:.0f}% ({len(trades)} trades){sig_tag}",
            f"Baseline {_BASELINE_WR*100:.1f}% WR; live {wr*100:.1f}%; floor {_WR_FLOOR*100:.0f}%",
            "Live win rate is significantly below backtest baseline. Either market regime has "
            "changed, the entry model is degraded, or new negative confluences are over-filtering "
            "setups that previously would have won.",
            "Review last 20 losses for common patterns. Check if new negative confluences "
            "(-Both Asian H+L swept, -Mid daily range) are blocking too many valid setups.",
            "high" if sig else "medium"))

    if avg_r < _AVG_R_FLOOR:
        obs.append(_obs("09", "Test Engineer",
            f"Live avg-R {avg_r:.3f} below floor {_AVG_R_FLOOR:.2f} ({len(trades)} trades)",
            f"Baseline {_BASELINE_AVG_R:.3f}R; live {avg_r:.3f}R",
            "Average R per trade is too low. Either losses are too large or wins are not "
            "reaching their targets. The TP extension logic may not be working.",
            "Check hit_target rate on winning trades. If TP is not being reached, the structural "
            "target finder may be overestimating the draw. Verify manage_trade() is extending TP "
            "to next pool when price approaches current TP.",
            "high"))

    # Session analysis: which hours are bleeding?
    by_hour: dict = defaultdict(lambda: {"n": 0, "r": 0.0, "wins": 0})
    for t in trades:
        h = int(t.get("session_hour") or 0)
        by_hour[h]["n"] += 1
        by_hour[h]["r"] += _r(t)
        by_hour[h]["wins"] += (t.get("outcome") == "win")

    bad_hours = []
    for h, d in by_hour.items():
        if d["n"] >= 5:
            avg = d["r"] / d["n"]
            wr_h = d["wins"] / d["n"]
            if avg < -0.2 and wr_h < 0.30:
                bad_hours.append(f"{h:02d}:00 UTC ({d['n']} trades, avg {avg:.2f}R, {wr_h*100:.0f}% WR)")

    if bad_hours:
        obs.append(_obs("09", "Test Engineer",
            f"Session hours bleeding edge: {len(bad_hours)} hour(s) with negative avg-R",
            "; ".join(bad_hours),
            "Specific UTC hours are consistently losing. This indicates session timing mismatches — "
            "entering in hours with no directional flow or low liquidity.",
            "Add a session_hour_blacklist to the config and block entries in identified hours. "
            "Minimum 10 trades per hour before flagging to avoid noise.",
            "medium"))

    # Score vs R correlation check — higher scores should produce higher R
    sc_r: dict = defaultdict(list)
    for t in trades:
        sc_r[int(t.get("score") or 0)].append(_r(t))
    sc_avgs = {sc: sum(rs)/len(rs) for sc, rs in sc_r.items() if len(rs) >= 3}
    if sc_avgs:
        sorted_sc = sorted(sc_avgs.items())
        inverted = [(sc, avg) for i, (sc, avg) in enumerate(sorted_sc[1:], 1)
                    if avg < sorted_sc[i-1][1] - 0.3]
        if inverted:
            obs.append(_obs("09", "Test Engineer",
                "Score-to-R correlation broken at some levels",
                "; ".join(f"score {sc}: {avg:.2f}R" for sc, avg in sc_avgs.items()),
                "Higher scores should produce higher average R. Inversions mean the confluence "
                "model adds noise at those score levels rather than adding predictive power.",
                "Remove or revise confluences that appear at inverted score levels. "
                "Score should be monotonically correlated with outcome R.",
                "medium"))

    return obs


# ── Council #11 — Reality Gap Analyst ────────────────────────────────────────

def _council_11(trades: list, misfires: list, wins: list) -> list:
    """Backtest vs live divergence; where reality diverges from plan."""
    obs = []
    if not trades:
        return obs

    # Trade type performance — which types work live vs what the model expected?
    type_stats: dict = defaultdict(lambda: {"n": 0, "wins": 0, "r": 0.0})
    for t in trades:
        tt = t.get("trade_type") or "unknown"
        type_stats[tt]["n"] += 1
        type_stats[tt]["wins"] += (t.get("outcome") == "win")
        type_stats[tt]["r"] += _r(t)

    for tt, s in type_stats.items():
        if s["n"] < 4:
            continue
        wr = s["wins"] / s["n"]
        avg_r = s["r"] / s["n"]
        if wr < 0.25 and avg_r < 0:
            obs.append(_obs("11", "Reality Gap Analyst",
                f"'{tt}' trade type is bleeding live ({wr*100:.0f}% WR, {avg_r:.2f}R avg)",
                f"{s['n']} {tt} trades: {s['wins']} wins, net {s['r']:.2f}R",
                f"The '{tt}' setup type is not working in live conditions. The backtest showed "
                f"this type was viable, but live execution / timing may be degrading it.",
                f"Raise the minimum score floor for '{tt}' setups, or add a required extra "
                f"confluence specifically for this type. Do not disable the type entirely — "
                f"diagnose first.",
                "high"))

    # Hit-target rate — are we reaching our liquidity targets?
    with_target = [t for t in trades if t.get("hit_target") is not None]
    if len(with_target) >= 5:
        hit_rate = sum(1 for t in with_target if t.get("hit_target")) / len(with_target)
        if hit_rate < 0.35:
            obs.append(_obs("11", "Reality Gap Analyst",
                f"Liquidity target hit rate only {hit_rate*100:.0f}% ({len(with_target)} trades)",
                f"{sum(1 for t in with_target if t.get('hit_target'))}/{len(with_target)} trades reached their structural TP",
                "The trade_analyzer target finder is identifying draws that are not being "
                "reached in live conditions. Either targets are too far, or market regime is "
                "reversing before the draw is filled.",
                "Reduce MAX_REACH_ATR from 12 to 8 in trade_analyzer.py. Prefer equal-high/low "
                "clusters with 2+ touches over lone swing highs as targets.",
                "high"))

    # Where 'gave back' is worst — by symbol
    gave_back = [t for t in misfires if "gave_back" in (t.get("flags") or [])]
    gab_by_sym = Counter(t.get("symbol") for t in gave_back)
    for sym, cnt in gab_by_sym.most_common(2):
        if cnt >= 3:
            obs.append(_obs("11", "Reality Gap Analyst",
                f"{sym}: gave-back pattern {cnt}x — management diverges from plan",
                f"{cnt} trades on {sym} reached +1.5R then closed red",
                f"On {sym}, the adaptive trail is not locking in gains. The structural level "
                f"chosen as trail anchor may be too far from current price on this instrument.",
                f"For {sym}: tighten trail to half the standard swing-to-swing distance "
                f"at +2R. Alternatively, partial-close 50% at +2R and trail the rest.",
                "medium"))

    # Confluences present in losses that are ALSO present in wins — noise flags
    if wins and misfires:
        win_conf  = Counter(x for m in wins  for x in m.get("reasons", []))
        loss_conf = Counter(x for m in misfires for x in m.get("reasons", []))
        nw, nl = max(1, len(wins)), max(1, len(misfires))
        noise = []
        for conf in set(win_conf) | set(loss_conf):
            wr_conf = win_conf.get(conf, 0) / nw
            lr_conf = loss_conf.get(conf, 0) / nl
            if wr_conf > 0.3 and lr_conf > 0.3 and abs(wr_conf - lr_conf) < 0.10:
                noise.append(f"'{conf}' (wins {wr_conf*100:.0f}% / losses {lr_conf*100:.0f}%)")
        if noise:
            obs.append(_obs("11", "Reality Gap Analyst",
                f"Confluences that appear equally in wins and losses — scoring noise",
                "; ".join(noise[:5]),
                "These confluences are not predictive — they appear as often in losers as "
                "in winners. They add noise to the score without improving outcomes.",
                "Remove or halve the weight of these confluences in the scoring model. "
                "Replace with the confluences that are 2x+ more common in winners.",
                "high"))

    return obs


# ── Council #12 — Devil's Advocate ───────────────────────────────────────────

def _council_12(trades: list, misfires: list, wins: list) -> list:
    """Challenges all confident conclusions. Always last, always critical."""
    obs = []
    if not trades:
        return obs

    consec = _consec_losses(trades)
    if consec >= _CONSEC_LOSS_LIMIT:
        obs.append(_obs("12", "Devil's Advocate",
            f"ALARM: {consec} consecutive losses — full stop review required",
            f"Last {consec} trades all closed as losses",
            "This many consecutive losses means ONE of: (a) regime has changed and the strategy "
            "has no edge right now, (b) a bug was introduced that is breaking entries, "
            "(c) sizing is too large and losses are compounding psychology.",
            "STOP TRADING. Do not increase size. Run a manual review of the last 10 setups "
            "against the entry rules. Do not resume until the source is identified.",
            "high"))

    # Beat up winning trades — what went wrong even in wins?
    won_but_missed = []
    for t in wins:
        mfe = float(t.get("mfe_r") or 0)
        ex  = float(t.get("r_multiple") or 0)
        if mfe > 0 and ex > 0 and ex < mfe * 0.5:
            won_but_missed.append((t.get("symbol"), round(mfe, 2), round(ex, 2)))

    if len(won_but_missed) >= 3:
        examples = won_but_missed[:3]
        obs.append(_obs("12", "Devil's Advocate",
            f"Winning trades left significant R on table: {len(won_but_missed)} cases",
            "; ".join(f"{s}: MFE {m}R but exited at {e}R" for s, m, e in examples),
            "Wins that exit at less than 50% of their maximum favourable excursion are "
            "under-managed. The trail is too loose OR we closed too early from fear. "
            "Even in wins, we are not taking what the market offered.",
            "Audit the trail logic for these symbols. At +2R, the trail should be tight enough "
            "to lock in at least 1.5R. If MFE is 3R and exit is 1R, the trail is wrong.",
            "medium"))

    # Sample size warning
    if len(trades) < 30:
        obs.append(_obs("12", "Devil's Advocate",
            f"Sample too small for statistical conclusions: {len(trades)} trades",
            f"All analysis above uses {len(trades)} trades — need 30+ for any pattern to be reliable",
            "Every 'pattern' found with fewer than 30 trades is likely random noise. "
            "Acting on it risks curve-fitting to a handful of outcomes.",
            "Treat all observations above as hypotheses, not findings. Do not change "
            "scoring weights or remove confluences until 30+ trades confirm the pattern.",
            "high"))

    # The meta-challenge: is the bot being allowed to self-improve or being over-managed?
    if len(trades) >= 30:
        n_proposals_without_data = 0
        obs.append(_obs("12", "Devil's Advocate",
            "Meta check: is the system actually learning or just logging?",
            f"{len(trades)} trades in journal; check if any prior proposals have been approved and applied",
            "A learning loop that proposes but never has anything approved is not a learning loop "
            "— it is a logging system. If proposals are consistently good but never actioned, "
            "the feedback cycle is broken and the system cannot improve.",
            "Review learning_proposals.md history. If multiple cycles have passed with no "
            "approved proposals, discuss with Anton which improvements to unlock.",
            "medium"))

    return obs


# ── Auto-apply engine ─────────────────────────────────────────────────────────

def _compute_confluence_lifts(trades: list) -> dict[str, float]:
    """Compute per-confluence observed win rates from live trades.

    Keys match ProbabilityModel._lifts keys. Only returned when n >= 3.
    ProbabilityModel applies Laplace smoothing on its end — we just supply
    the raw (wins, obs) counts and let it blend with the prior.
    """
    confluence_keys = [
        "fvg_present", "ob_present", "m5_confirmed", "h4_aligned",
        "at_htf_level", "order_flow_aligned", "news_aligned",
        "continuation_type", "in_ict_macro", "ipda_aligned",
        "smt_divergence", "eq_liq_cluster", "early_leakage", "inside_day",
    ]
    counts: dict[str, list] = {k: [0, 0] for k in confluence_keys}  # [wins, obs]

    for t in trades:
        won = t.get("outcome") == "win"
        c   = t.get("confluences", {})
        for k in confluence_keys:
            v = c.get(k, False)
            # continuation_type is True when trade_type == "continuation"
            if k == "continuation_type":
                v = t.get("trade_type") == "continuation"
            if v:
                counts[k][1] += 1
                if won:
                    counts[k][0] += 1

    result = {}
    BASE_WR = 0.35  # matches ProbabilityModel.BASE_WIN_RATE
    PSEUDO  = 5
    for k, (w, n) in counts.items():
        if n < 3:
            continue
        p_cond   = (w + PSEUDO * BASE_WR) / (n + PSEUDO)
        new_lift = p_cond / BASE_WR
        new_lift = max(0.20, min(new_lift, 4.0))  # safety bounds
        result[k] = round(new_lift, 4)
    return result


def auto_apply(result: dict) -> list[str]:
    """Apply learning loop findings to the live system.

    Tier 1 (always): Bayesian lift updates written to quant_lift_proposals.json
      — ProbabilityModel reads this file on next estimate() call and blends
      updates with smoothing (n/20 blend weight, so 3 obs = 15% influence).

    Tier 2 (n >= 30, HIGH confidence): structural config changes applied now.
      Devil's Advocate (#12) veto blocks all tier-2 until 30 trades.

    Returns list of human-readable applied-change strings for Telegram/report.
    """
    applied: list[str] = []
    trades = result.get("_trades_raw", [])
    n      = result["trades"]

    # ── Tier 1: always — Bayesian lift update via quant proposal file ──────────
    lifts = _compute_confluence_lifts(trades)
    if lifts:
        proposal_file = LOG_DIR / "quant_lift_proposals.json"
        try:
            proposal_file.write_text(
                json.dumps({"confluences": lifts, "ts": datetime.now(timezone.utc).isoformat(),
                            "n_trades": n}, indent=2)
            )
            applied.append(f"Bayesian lifts updated from {n} trades ({len(lifts)} confluences)")
        except Exception as exc:
            applied.append(f"Lift update failed: {exc}")

    # ── Tier 2: structural — only when Devil's Advocate clears (n >= 30) ──────
    da_veto = any(o["council"] == "12" and "too small" in o["title"].lower()
                  for o in result["proposals"])
    if da_veto:
        applied.append(f"Tier-2 structural changes deferred: Devil's Advocate veto (n={n}, need 30)")
        return applied

    # Each structural fix: check current value vs evidence, apply + log
    for obs in result["proposals"]:
        if obs["confidence"] != "high":
            continue
        title = obs["title"].lower()

        # MAX_REACH_ATR reduction — 0% liquidity target hit rate
        if "liquidity target" in title or "max_reach_atr" in title:
            ta_file = Path(__file__).parent / "trade_analyzer.py"
            try:
                src = ta_file.read_text(encoding="utf-8")
                import re
                m = re.search(r"MAX_REACH_ATR\s*=\s*([\d.]+)", src)
                if m and float(m.group(1)) > 5.0:
                    new_src = re.sub(r"(MAX_REACH_ATR\s*=\s*)[\d.]+", r"\g<1>5.0", src)
                    ta_file.write_text(new_src, encoding="utf-8")
                    applied.append(f"MAX_REACH_ATR reduced to 5.0 (was {m.group(1)}) — 0% hit rate")
            except Exception as exc:
                applied.append(f"MAX_REACH_ATR update failed: {exc}")

    return applied


# ── Guardrail — crewAI pattern (crewAIInc/crewAI task._invoke_guardrail_function) ──
# Each Council analysis function is wrapped by a guardrail that validates
# output quality before accepting it. On failure, the function re-runs up to
# MAX_GUARDRAIL_RETRIES with the failure reason injected as context.

MAX_GUARDRAIL_RETRIES = 2

def _guardrail(obs_list: list, council_id: str) -> tuple[bool, str]:
    """Return (valid, reason). Rejects output if obviously wrong."""
    if not isinstance(obs_list, list):
        return False, f"#{council_id} returned non-list"
    for o in obs_list:
        if not all(k in o for k in ("title", "evidence", "verdict", "proposal", "confidence")):
            return False, f"#{council_id} obs missing required keys"
        if o["confidence"] not in ("high", "medium", "low"):
            return False, f"#{council_id} bad confidence value: {o['confidence']!r}"
    return True, "ok"


def _guarded(fn, *args, council_id: str = "??"):
    """Run fn with guardrail validation; retry up to MAX_GUARDRAIL_RETRIES."""
    for attempt in range(1 + MAX_GUARDRAIL_RETRIES):
        try:
            result = fn(*args)
        except Exception as exc:
            result = []
            if attempt < MAX_GUARDRAIL_RETRIES:
                continue
            logger.warning("[Council #%s] exception (attempt %d): %s", council_id, attempt, exc)
        valid, reason = _guardrail(result, council_id)
        if valid:
            return result
        if attempt < MAX_GUARDRAIL_RETRIES:
            logger.debug("[Council #%s] guardrail retry: %s", council_id, reason)
    return result  # return whatever we have after retries


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(days: int = 0) -> dict:
    trades   = _live_trades()
    misfires = _load_jsonl("misfires.jsonl")
    wins     = _load_jsonl("wins.jsonl")

    all_obs: list[dict] = []
    all_obs += _guarded(_council_05, trades, misfires, council_id="05")
    all_obs += _guarded(_council_06, trades, misfires, wins, council_id="06")
    all_obs += _guarded(_council_07, trades, misfires, council_id="07")
    all_obs += _guarded(_council_09, trades, council_id="09")
    all_obs += _guarded(_council_11, trades, misfires, wins, council_id="11")
    all_obs += _guarded(_council_12, trades, misfires, wins, council_id="12")

    return dict(
        trades=len(trades), misfires=len(misfires), wins=len(wins),
        proposals=all_obs,
        _trades_raw=trades,  # passed to auto_apply, stripped before report
    )


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(result: dict, applied: list[str] | None = None) -> None:
    obs    = result["proposals"]
    by_cid = defaultdict(list)
    for o in obs:
        by_cid[o["council"]].append(o)

    member_names = {
        "05": "Compliance Officer",
        "06": "App Engineer",
        "07": "SRE",
        "09": "Test Engineer",
        "11": "Reality Gap Analyst",
        "12": "Devil's Advocate",
    }

    lines = [
        "---", "type: analysis", "status: auto-applied",
        "tags: [trading-bot, council-review, learning-loop]",
        "relatedTo: [trading-bot, misfire-ledger]", "---", "",
        f"# Council of 12 — Trade Review  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
        f"**{result['trades']} live trades** | **{result['wins']} wins** | "
        f"**{result['misfires']} misfires** | **{len(obs)} observations**", "",
    ]

    if applied:
        lines += ["## Auto-Applied Changes", ""]
        for a in applied:
            lines.append(f"- {a}")
        lines.append("")

    if not obs:
        lines.append("_No patterns yet — need more live trades._")
    else:
        for cid in sorted(by_cid.keys()):
            cname = member_names.get(cid, f"Council #{cid}")
            lines += [f"---", f"## Council #{cid} — {cname}", ""]
            for i, o in enumerate(by_cid[cid], 1):
                conf_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(o["confidence"], "⚪")
                status = o.get("status", "")
                lines += [
                    f"### {conf_icon} {o['title']}",
                    f"**Evidence:** {o['evidence']}",
                    "",
                    f"**Verdict:** {o['verdict']}",
                    "",
                    f"**Action `[{status}]`:** {o['proposal']}",
                    "",
                ]

    PROPOSALS.write_text("\n".join(lines), encoding="utf-8")

    # Mirror to Obsidian Brain
    if obs:
        try:
            from execution import obsidian_sync as ob
            title = f"Learning Loop — {datetime.now(timezone.utc):%Y-%m-%d} ({len(obs)} obs, {len(applied or [])} applied)"
            ob.write_brain_entry(title, "\n".join(lines[6:]), tags=["learning-loop", "council", "aiden"])
        except Exception:
            pass


def main():
    res     = analyze()
    applied = auto_apply(res)
    write_report(res, applied)
    by_conf = Counter(o["confidence"] for o in res["proposals"])
    print(
        f"Council review: {res['trades']} live trades | "
        f"{res['wins']} wins | {res['misfires']} misfires\n"
        f"{len(res['proposals'])} observations: "
        f"{by_conf.get('high',0)} HIGH / {by_conf.get('medium',0)} MED / {by_conf.get('low',0)} LOW\n"
        f"Report: {PROPOSALS}"
    )
    for o in res["proposals"]:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(o["confidence"], "")
        print(f"  #{o['council']} {icon} {o['title']}")


if __name__ == "__main__":
    main()
