"""Cadre agent invocation — runs a Cadre member autonomously via claude -p.

Called by council_watch when a judgment-call event occurs. Each Cadre member
(Sage/Quant/Builder/Scout) has a defined scope and prompt template.

Usage:
    python -m execution.cadre_invoke --member MEM-002 --event daily_loss_review
    python -m execution.cadre_invoke --member MEM-001 --event weekly_strategy_review
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_BOT_ROOT  = Path(__file__).resolve().parent.parent
_LOGS      = _BOT_ROOT / "logs"
_FIRM_DB   = Path(r"C:\Users\anton\.firm\firm.db")
_VAULT     = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden\AiDEN")

logger = logging.getLogger("cadre_invoke")

CADRE_SCOPE = {
    "MEM-001": {  # Sage — Strategist
        "name": "Sage",
        "model": "claude-fable-5",   # strategy reasoning needs the top model
        "triggers": ["weekly_strategy_review", "regime_shift", "strategy_evolution"],
        "prompt_template": (
            "You are Sage (MEM-001), AiDEN's Strategist. Your role: high-level trading strategy, "
            "system design, and regime analysis.\n\n"
            "Event: {event}\nContext: {context}\n\n"
            "Review the current strategy state. Produce: (1) regime assessment, (2) any strategy "
            "adjustments needed, (3) updated rules for the Brain vault. "
            "Write your output to C:\\Users\\anton\\OneDrive\\Desktop\\Aiden\\AiDEN\\Brain\\ "
            "as a new insight note with MOP frontmatter."
        ),
    },
    "MEM-002": {  # Quant — Analyst
        "name": "Quant",
        "triggers": ["post_incident_review", "loss_streak_5", "win_rate_drift", "daily_loss_review"],
        "prompt_template": (
            "You are Quant (MEM-002), AiDEN's Quantitative Analyst. Your role: data analysis, "
            "backtest validation, strategy statistics, and post-incident reviews.\n\n"
            "Event: {event}\nContext: {context}\n\n"
            "Analyze the trading data at C:\\Users\\anton\\Documents\\trading-bot\\logs\\. "
            "Produce a structured analysis. Write your findings to the Obsidian Brain vault "
            "at C:\\Users\\anton\\OneDrive\\Desktop\\Aiden\\AiDEN\\Brain\\ with MOP frontmatter."
        ),
    },
    "MEM-003": {  # Builder — Engineer
        "name": "Builder",
        "model": "sonnet",   # autonomous code edits need more than haiku
        "triggers": ["code_fix_needed", "bot_error_detected", "performance_issue"],
        "prompt_template": (
            "You are Builder (MEM-003), AiDEN's Engineer. Your role: bot code, execution logic, "
            "bug fixes, and system improvements.\n\n"
            "Event: {event}\nContext: {context}\n\n"
            "Investigate and fix the issue in C:\\Users\\anton\\Documents\\trading-bot\\. "
            "Apply the minimal targeted fix. Log what you changed to the Obsidian Brain vault."
        ),
    },
    "MEM-004": {  # Scout — Researcher
        "name": "Scout",
        "triggers": ["market_regime_research", "news_event", "instrument_analysis"],
        "prompt_template": (
            "You are Scout (MEM-004), AiDEN's Researcher. Your role: market intelligence, "
            "news analysis, instrument research, and external signal gathering.\n\n"
            "Event: {event}\nContext: {context}\n\n"
            "Research the event and context. Produce a brief intelligence report. "
            "Write it to C:\\Users\\anton\\OneDrive\\Desktop\\Aiden\\AiDEN\\Brain\\ with MOP frontmatter."
        ),
    },
}


def _build_context(event: str) -> str:
    ctx_parts = []

    # Attach recent session note
    sessions = sorted(_VAULT.glob("Sessions/*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if sessions:
        ctx_parts.append(f"Last session note ({sessions[0].name}):\n{sessions[0].read_text(encoding='utf-8')[:1000]}")

    # Attach ftmo tracker state
    ftmo_state = _LOGS / "ftmo_tracker_state.json"
    if ftmo_state.exists():
        ctx_parts.append(f"FTMO state:\n{ftmo_state.read_text(encoding='utf-8')}")

    # Attach recent council events
    events_log = _LOGS / "council_events.jsonl"
    if events_log.exists():
        lines = events_log.read_text(encoding="utf-8").strip().splitlines()[-10:]
        ctx_parts.append(f"Recent council events:\n" + "\n".join(lines))

    return "\n\n---\n\n".join(ctx_parts)


def invoke(member_id: str, event: str, extra_context: str = "") -> int:
    member = CADRE_SCOPE.get(member_id)
    if member is None:
        logger.error("Unknown Cadre member: %s", member_id)
        return 1

    context = _build_context(event)
    if extra_context:
        context = extra_context + "\n\n" + context

    prompt = member["prompt_template"].format(event=event, context=context[:6000])
    # Scout's persona idles ("ready for work") unless told the task IS this
    # invocation — observed 2026-07-27: task in Context: block was ignored.
    prompt += (
        "\n\nIMPORTANT: The task above is assigned to THIS invocation. Execute it NOW — "
        "do not report 'ready' or wait for a queued task. Write every required output "
        "file to the exact paths given before finishing."
    )

    # Find claude CLI
    claude_candidates = [
        Path(r"C:\Users\anton\AppData\Roaming\npm\claude.cmd"),
        Path(r"C:\Program Files\nodejs\claude.cmd"),
    ]
    claude_cmd = None
    for c in claude_candidates:
        if c.exists():
            claude_cmd = str(c)
            break

    if claude_cmd is None:
        # Try PATH
        claude_cmd = "claude"

    model = member.get("model", "haiku")
    # Headless -p runs get no permission prompts — without explicit tool
    # grants the agent cannot write its output files and the feedback loop
    # silently dies at the last step.
    # Prompt goes via stdin: claude.cmd routes through cmd.exe on Windows,
    # which truncates multi-line argv — Scout was only seeing line 1 of its
    # persona and idling with "ready for work".
    cmd = [
        claude_cmd, "-p", "--model", model,
        "--allowedTools", "Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch",
    ]

    logger.info("[Cadre] Invoking %s (%s) for event: %s", member["name"], member_id, event)

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            cwd=str(_BOT_ROOT),
        )
        if result.returncode == 0:
            logger.info("[Cadre] %s completed: %s", member["name"], result.stdout[:200])
            # Log the invocation
            _log_invocation(member_id, event, result.stdout, success=True)
            return 0
        else:
            logger.error("[Cadre] %s failed (exit %d): %s", member["name"], result.returncode, result.stderr[:200])
            _log_invocation(member_id, event, result.stderr, success=False)
            return 1
    except subprocess.TimeoutExpired:
        logger.error("[Cadre] %s timed out for event: %s", member["name"], event)
        return 1
    except FileNotFoundError:
        logger.error("[Cadre] claude CLI not found — Cadre invocations require Claude Code CLI installed")
        return 1


def _log_invocation(member_id: str, event: str, output: str, success: bool) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "cadre_invocation",
        "member": member_id,
        "event": event,
        "success": success,
        "output_preview": output[:500],
    }
    events_log = _LOGS / "council_events.jsonl"
    with events_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True, choices=list(CADRE_SCOPE.keys()))
    parser.add_argument("--event", required=True)
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    sys.exit(invoke(args.member, args.event, args.context))
