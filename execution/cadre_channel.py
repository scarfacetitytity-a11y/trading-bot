"""CadreChannel — Agent-Reach multi-backend dispatch layer for the AiDEN Cadre.

Wraps the four Cadre agents (Sage/Quant/Builder/Scout) with the same ordered-
backend fallback pattern used in news_gate.py and scout_reach.py. Each agent is
a "backend". When an invocation fails, dispatch() routes to the next available
agent in the channel's backend list — first success wins.

This module is pure Python dispatch. It does not call LLMs directly; it delegates
to cadre_invoke.invoke(), which spawns the Claude CLI. The handler is injectable
so callers can substitute a stub for testing without touching actual LLM calls.

Usage:
    from execution.cadre_channel import CadreChannel, ANALYSTS, ALL_CADRE

    channel = CadreChannel("analysis", backends=ANALYSTS)
    result  = channel.dispatch({"event": "daily_loss_review", "context": ""})
    if result["status"] == "ok":
        print(result["agent"], "handled the task")
    else:
        print("All agents failed:", result["errors"])
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Canonical Cadre member IDs — matches cadre_invoke.CADRE_SCOPE keys.
SAGE    = "MEM-001"
QUANT   = "MEM-002"
BUILDER = "MEM-003"
SCOUT   = "MEM-004"

# Pre-built backend lists for common routing needs.
ANALYSTS  = [QUANT, SAGE]            # data/review tasks; Quant primary, Sage escalation
ENGINEERS = [BUILDER, SAGE]          # code tasks; Builder primary, Sage escalation
INTEL     = [SCOUT, QUANT]           # research tasks; Scout primary, Quant fallback
ALL_CADRE = [SAGE, QUANT, BUILDER, SCOUT]  # full council fallback


def _default_invoke(agent_id: str, task: dict) -> dict:
    """Default handler: delegates to cadre_invoke.invoke() and adapts the int exit code.

    cadre_invoke.invoke() returns 0 on success, non-zero on failure — it writes
    output to Obsidian/logs rather than returning it. We wrap that int into the
    dict contract CadreChannel expects.
    """
    from execution import cadre_invoke  # local import avoids circular dep at module load

    event   = task.get("event", "unknown_event")
    context = task.get("context", "")
    rc      = cadre_invoke.invoke(agent_id, event, extra_context=context)
    if rc == 0:
        return {"status": "ok", "agent": agent_id, "event": event}
    raise RuntimeError(f"cadre_invoke returned exit code {rc} for {agent_id}/{event}")


class CadreChannel:
    """Ordered-backend dispatch channel for Cadre agent routing.

    Attributes:
        name:     Logical channel name (for logging).
        backends: Ordered list of Cadre agent IDs to try. First success wins.
        _handler: Callable(agent_id, task) -> dict. Injectable for testing.
    """

    def __init__(
        self,
        name: str,
        backends: list[str],
        handler: Optional[Callable[[str, dict], dict]] = None,
    ) -> None:
        self.name     = name
        self.backends = list(backends)
        self._handler = handler or _default_invoke

    def probe(self, agent_id: str) -> bool:
        """Check whether an agent is registered and available for dispatch.

        Cheap static check — never spawns an LLM call. Returns False for
        unknown agent IDs so dispatch() can skip them gracefully.
        """
        from execution import cadre_invoke
        return agent_id in cadre_invoke.CADRE_SCOPE

    def dispatch(self, task: dict) -> dict:
        """Try each backend in order; return the first successful result.

        Args:
            task: Dict with at minimum {"event": str}. Optional "context" str.

        Returns:
            On success: {"status": "ok", "agent": agent_id, ...handler output...}
            On total failure: {"status": "error", "errors": {agent_id: reason, ...}}
        """
        errors: dict[str, str] = {}

        for agent_id in self.backends:
            if not self.probe(agent_id):
                reason = f"agent {agent_id} not registered"
                logger.warning("[CadreChannel:%s] %s — skipping", self.name, reason)
                errors[agent_id] = reason
                continue

            try:
                result = self._handler(agent_id, task)
                logger.info("[CadreChannel:%s] %s handled task '%s'", self.name, agent_id, task.get("event"))
                return result
            except Exception as exc:
                reason = str(exc)
                logger.warning(
                    "[CadreChannel:%s] %s failed: %s — trying next backend",
                    self.name, agent_id, reason,
                )
                errors[agent_id] = reason

        logger.error(
            "[CadreChannel:%s] All backends failed for task '%s': %s",
            self.name, task.get("event"), errors,
        )
        return {"status": "error", "errors": errors, "event": task.get("event")}
