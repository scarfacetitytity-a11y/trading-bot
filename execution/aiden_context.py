"""AiDEN codebase context builder.

Generates a structured system prompt describing the AiDEN trading bot
for use when handing off to external AI models (Groq, Gemini, Kimi, etc.)
when Claude hits token/usage limits.

Includes: architecture, active files, recent commits, FTMO state, open trades.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def _git(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=_ROOT, stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace"
        ).strip()
    except Exception:
        return ""


def _read_safe(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[:max_lines])
    except Exception:
        return ""


def _ftmo_state() -> str:
    state_file = _ROOT / "logs" / "risk_agent_state.json"
    try:
        d = json.loads(state_file.read_text(encoding="utf-8"))
        equity = d.get("current_equity", "?")
        daily_dd = d.get("daily_loss_pct", "?")
        total_dd = d.get("total_loss_pct", "?")
        return f"equity={equity} daily_dd={daily_dd}% total_dd={total_dd}%"
    except Exception:
        return "state unavailable"


def _open_trades() -> str:
    trades_file = _ROOT / "logs" / "open_trades.json"
    try:
        trades = json.loads(trades_file.read_text(encoding="utf-8"))
        if not trades:
            return "none"
        return ", ".join(
            f"{t.get('symbol')} {'+' if t.get('direction',0)>0 else '-'}{t.get('lots','?')}L"
            for t in trades.values()
        )
    except Exception:
        return "unavailable"


def _recent_commits(n: int = 8) -> str:
    return _git(f"git log --oneline -{n}")


def _recent_changes() -> str:
    return _git("git diff HEAD~1 --stat")


def build_system_prompt(include_recent_code: bool = False) -> str:
    """Build a full AiDEN context system prompt for external AI models."""

    recent_commits = _recent_commits()
    ftmo = _ftmo_state()
    open_trades = _open_trades()

    prompt = f"""You are assisting with AiDEN — an autonomous FTMO $100k prop firm trading bot written in Python.

== SYSTEM ARCHITECTURE ==
- Orchestrator: execution/orchestrator.py — manages all per-symbol SignalEngines
- Strategy: strategies/aiden_index.py — FVG+OB confluence scoring /10 on M15/H1
- Risk: execution/risk_agent.py — FTMO daily 5% / total 10% DD enforcement
- Probability model: execution/probability_model.py — Bayesian win-probability per trade
- Trade journal: execution/trade_journal.py — durable trade state, importance scoring
- Learning loop: execution/learning_loop.py — auto-applies Bayesian lift updates nightly
- Council governance: council/council_router.py — OmniRoute-adapted persona routing
- LLM advisor: execution/llm_advisor.py — multi-model AI advisory (you are one of the models)
- Event bus: execution/aiden_event_bus.py — cross-process JSONL event stream
- Watchdog: execution/watchdog.py — process supervisor with heartbeat
- Scout: execution/scout_reach.py — daily market brief generator

== INSTRUMENTS ==
XAUUSD, XAGUSD, GBPUSD, EURUSD, US30.cash, US100.cash, US500.cash, US2000.cash, JP225.cash
All BIDIRECTIONAL (long + short). Minimum entry score: 7/10.

== FTMO RULES ==
- Daily drawdown limit: 5% of starting equity (HARD — bot halts at 4.5%)
- Max total drawdown: 10% (HARD stop)
- Profit target: 10% to pass Phase 1
- Risk per trade: fixed-fractional ~0.75% equity
- allow_real_account: false — demo account ONLY, never change this

== LIVE STATE ==
FTMO: {ftmo}
Open trades: {open_trades}

== RECENT WORK (last 8 commits) ==
{recent_commits}

== KEY RULES ==
- Never set long_only=True anywhere — fully bidirectional always
- No naked positions — every trade must have a stop loss
- Score gate: signal_score < needed → skip, never override
- Council #05 (Compliance) + #12 (Devil's Advocate) always run on every trade
- Learning loop auto-applies Bayesian lifts after every trade close (two-tier: Tier1 always, Tier2 at n≥30)

== HOW TO HELP ==
You are a senior Python engineer and quant trader working on this codebase.
When asked to write code: match the existing style (no comments explaining what code does, no docstrings unless complex, type hints, pathlib not os.path).
When asked about trading decisions: apply FTMO rules strictly. Devil's Advocate always gets last word.
When asked to debug: read the relevant file first, fix the root cause, never paper over with try/except.
"""

    if include_recent_code:
        diff = _git("git diff HEAD~1 -- execution/ council/ strategies/ core/")
        if diff:
            prompt += f"\n== RECENT CODE CHANGES ==\n{diff[:3000]}\n"

    return prompt.strip()


def export_handoff(output_path: Path | None = None) -> str:
    """Generate a handoff document — paste into any AI to resume work."""
    system = build_system_prompt(include_recent_code=True)

    handoff = f"""AIDEN HANDOFF DOCUMENT — {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
Copy everything below this line and paste as your first message to the AI model.
{'='*70}

{system}

== TASK ==
[Describe what you were working on when Claude ran out of tokens, or ask your question below.]
"""

    if output_path:
        output_path.write_text(handoff, encoding="utf-8")

    return handoff
