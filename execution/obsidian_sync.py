"""AiDEN → Obsidian brain sync.

Writes structured notes into the Obsidian vault so every trade, session,
council verdict, and system state is accessible from the graph view.

Folder layout inside the vault:
  AiDEN/
    Trades/        — one note per closed trade
    Sessions/      — daily session summary (updated on close)
    Council/       — learning_proposals snapshots
    System/        — bot state, key levels, instrument snapshots
    Brain/         — persistent facts the bot has internalized

Graph links:  every note links to its [[Symbol]], [[Session]], [[Outcome]]
so Obsidian graph shows patterns across confluences and instruments.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VAULT = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden")
AIDEN = VAULT / "AiDEN"


def _ensure(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    logger.info("[Obsidian] wrote %s", path.relative_to(VAULT))


# ── Trade note ────────────────────────────────────────────────────────────────

def write_trade(trade: dict) -> None:
    """Write one closed trade as an Obsidian note with full graph links."""
    sym       = trade.get("symbol", "UNKNOWN")
    direction = "LONG" if trade.get("direction", 1) == 1 else "SHORT"
    outcome   = trade.get("outcome", "unknown").upper()
    r         = trade.get("r_multiple", 0.0) or 0.0
    score     = trade.get("score", 0)
    open_t    = trade.get("open_time", "")[:16].replace("T", " ")
    close_t   = trade.get("close_time", "")[:16].replace("T", " ")
    pnl       = trade.get("pnl_usd", 0.0) or 0.0
    thesis    = trade.get("thesis", "")
    lesson    = trade.get("lesson", "")
    reasons   = trade.get("reasons", [])
    mfe       = trade.get("mfe_r", 0.0) or 0.0
    mae       = trade.get("mae_r", 0.0) or 0.0
    lots      = trade.get("lots", 0.0)
    entry     = trade.get("entry_price", 0.0)
    sl        = trade.get("sl_price", 0.0)
    tp        = trade.get("tp_price")
    grade     = trade.get("grade", "?")

    outcome_icon = {"WIN": "✅", "LOSS": "❌", "BREAKEVEN": "➖"}.get(outcome, "❓")
    date_str  = open_t[:10]
    slug      = f"{date_str}_{sym}_{direction}_{outcome}"

    reasons_md = "\n".join(f"- {r}" for r in reasons) if reasons else "- (none logged)"

    tp_val = f"{tp:.5g}" if tp else ""
    content = f"""---
type: trade
status: closed
tags: [trade, {sym.lower().replace('.','')}, {direction.lower()}, {outcome.lower()}, score-{score}]
relatedTo: [AiDEN, {sym}, {direction}, {outcome}]
date: {date_str}
symbol: {sym}
direction: {direction}
outcome: {outcome}
score: {score}
grade: {grade}
r_multiple: {r}
pnl_usd: {round(pnl, 2)}
lots: {lots}
entry_price: {entry}
sl_price: {sl}
tp_price: {tp_val}
mfe_r: {mfe}
mae_r: {mae}
open_time: "{open_t}"
close_time: "{close_t}"
---

# {outcome_icon} {direction} {sym} — {outcome} ({r:+.2f}R)

## Summary
| Field | Value |
|---|---|
| Symbol | [[{sym}]] |
| Direction | [[{direction}]] |
| Outcome | [[{outcome}]] |
| Score | {score}/10 |
| Grade | {grade} |
| R-Multiple | `{r:+.2f}R` |
| P&L | `${pnl:+,.2f}` |
| Lots | {lots} |
| Entry | `{entry:.5g}` |
| SL | `{sl:.5g}` |
| TP | {f"`{tp:.5g}`" if tp else "—"} |
| Open | {open_t} UTC |
| Close | {close_t} UTC |
| MFE | {mfe:.2f}R |
| MAE | {mae:.2f}R |

## Thesis
{thesis}

## Confluences
{reasons_md}

## Post-Trade Review
**Lesson:** {lesson}

**MFE {mfe:.2f}R / MAE {mae:.2f}R** — {"price reached {:.0f}% of target before reversing".format(mfe / ((abs(r) or 1)) * 100) if not trade.get("hit_target") else "target hit"}

## Links
[[{sym}]] | [[{direction}]] | [[{outcome}]] | [[{date_str}]] | [[Session {open_t[11:13]}h UTC]]
"""

    folder = _ensure(AIDEN / "Trades" / sym)
    _write(folder / f"{slug}.md", content)


# ── Daily session summary ─────────────────────────────────────────────────────

def write_session(date_str: str, trades: list[dict], equity: float, session_pnl: float) -> None:
    """Update today's session note — called after each trade close."""
    wins  = sum(1 for t in trades if t.get("outcome") == "win")
    loss  = sum(1 for t in trades if t.get("outcome") == "loss")
    be    = sum(1 for t in trades if t.get("outcome") == "breakeven")
    r_sum = sum(t.get("r_multiple", 0) or 0 for t in trades)
    pnl_icon = "🟢" if session_pnl >= 0 else "🔴"

    trade_rows = ""
    for t in trades:
        sym = t.get("symbol", "")
        d   = "L" if t.get("direction") == 1 else "S"
        oc  = t.get("outcome", "?")[:2].upper()
        r   = t.get("r_multiple", 0) or 0
        trade_rows += f"| [[{sym}]] | {d} | {oc} | `{r:+.2f}R` |\n"

    content = f"""---
type: session
status: active
tags: [session, daily, aiden]
relatedTo: [AiDEN, Trading]
date: {date_str}
---

# {pnl_icon} Session {date_str}

## P&L
| Metric | Value |
|---|---|
| Equity | `${equity:,.2f}` |
| Session P&L | `${session_pnl:+,.2f}` |
| Total R | `{r_sum:+.2f}R` |
| Wins / Losses / BE | {wins} / {loss} / {be} |

## Trades
| Symbol | Dir | Result | R |
|---|---|---|---|
{trade_rows}
## Links
[[AiDEN]] | [[Trading]]
"""
    folder = _ensure(AIDEN / "Sessions")
    _write(folder / f"{date_str}.md", content)


# ── Council proposals snapshot ────────────────────────────────────────────────

def write_council_snapshot(proposals_md: str) -> None:
    """Copy latest learning_proposals.md into vault as a dated council note."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder   = _ensure(AIDEN / "Council")
    dest     = folder / f"{date_str}-council-review.md"
    _write(dest, proposals_md)


# ── System state / key levels ─────────────────────────────────────────────────

def write_system_state(symbols: list[str], equity: float, brief: Optional[dict] = None) -> None:
    """Write current bot state + key levels into System/state.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    levels_md = ""
    if brief and brief.get("levels"):
        for lv in brief["levels"]:
            levels_md += f"- **{lv['symbol']}** {lv['label']} `{lv['price']:.5g}` ({lv.get('note','')})\n"

    sym_links = " | ".join(f"[[{s}]]" for s in symbols)

    content = f"""---
type: system-state
status: live
tags: [aiden, system, levels]
relatedTo: [AiDEN]
updated: {now}
---

# AiDEN System State

**Updated:** {now}
**Equity:** `${equity:,.2f}`
**Instruments:** {sym_links}

## Key Levels
{levels_md or "_No levels loaded_"}

## Links
[[AiDEN]]
"""
    folder = _ensure(AIDEN / "System")
    _write(folder / "state.md", content)


# ── Brain — persistent internalized facts ─────────────────────────────────────

def write_brain_entry(title: str, body: str, tags: list[str] | None = None) -> None:
    """Write a persistent insight into the Brain folder."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug     = title.lower().replace(" ", "-")[:40]
    tag_str  = ", ".join(tags or ["aiden", "insight"])
    content  = f"""---
type: insight
status: active
tags: [{tag_str}]
relatedTo: [AiDEN, Brain]
date: {date_str}
---

# {title}

{body}

---
_Internalized {date_str} by AiDEN_
"""
    folder = _ensure(AIDEN / "Brain")
    _write(folder / f"{date_str}-{slug}.md", content)


# ── Index note (vault entry point) ───────────────────────────────────────────

def write_index() -> None:
    """Write/update AiDEN/README.md — the graph entry point."""
    content = """---
type: index
status: active
tags: [aiden, index]
relatedTo: [AiDEN]
---

# AiDEN Brain

Central node for all AiDEN trading intelligence.

## Folders
- [[Trades/]] — every closed trade, linked by symbol + outcome
- [[Sessions/]] — daily P&L summaries
- [[Council/]] — Council of 12 review snapshots
- [[System/]] — live bot state, key levels
- [[Brain/]] — internalized lessons and patterns

## Quick Links
- [[System/state]] — current bot state
- [[LONG]] — all long trades
- [[SHORT]] — all short trades
- [[WIN]] — all wins
- [[LOSS]] — all losses
"""
    _ensure(AIDEN)
    _write(AIDEN / "README.md", content)
