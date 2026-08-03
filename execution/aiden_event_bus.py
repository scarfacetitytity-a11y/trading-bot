"""AiDEN Event Bus — cross-process event stream for the 4 autonomous processes.

Adapted from crewAI's crewai_event_bus pattern (crewAIInc/crewAI).
Events are appended to an append-only JSONL file (logs/aiden_events.jsonl).
Each autonomous process tracks its own read offset so it only processes
new events since last poll — no polling the full file, no shared memory.

Event types:
  TRADE_EXECUTED   — bot opened a new position
  TRADE_CLOSED     — position closed (win/loss + R multiple)
  RISK_LIMIT_NEAR  — daily DD approaching threshold
  BOT_RESTARTED    — watchdog restarted the bot
  LEARNING_APPLIED — auto_apply() ran and updated lifts/config
  SCOUT_BRIEF_READY — Scout wrote today's market brief
  COUNCIL_VERDICT  — CouncilRouter governance check result (approved/vetoed + personas)
  COUNCIL_VETO     — Council member flagged a trade for human review
  PERSONA_EXCLUDED — SelfHealingManager excluded a persona (backoff active)

Publishers (append_event):  orchestrator, watchdog, learning_loop, scout_reach
Subscribers (iter_new):     watchdog, code_monitor, council_watch, scout_reach

Usage (publisher):
    from execution.aiden_event_bus import append_event
    append_event("TRADE_EXECUTED", symbol="XAUUSD", direction=1, lots=0.10, score=72)

Usage (subscriber):
    from execution.aiden_event_bus import EventBusReader
    reader = EventBusReader("code_monitor")
    for event in reader.iter_new():
        handle(event)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_BUS_FILE  = Path(__file__).parent.parent / "logs" / "aiden_events.jsonl"
_STATE_DIR = Path(__file__).parent.parent / "logs"

# ── Publisher ─────────────────────────────────────────────────────────────────

def append_event(event_type: str, **payload) -> None:
    """Append one event to the bus. Fire-and-forget; never raises."""
    try:
        _BUS_FILE.parent.mkdir(exist_ok=True)
        record = {"ts": time.time(), "type": event_type, **payload}
        with _BUS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("[EventBus] append failed: %s", exc)


# ── Subscriber ────────────────────────────────────────────────────────────────

class EventBusReader:
    """Stateful reader for one subscriber process.

    Persists read offset to logs/{name}_bus_offset.txt so restarts pick up
    where they left off instead of re-processing old events.
    """

    def __init__(self, name: str) -> None:
        self._name       = name
        self._offset_file = _STATE_DIR / f"{name}_bus_offset.txt"
        self._offset      = self._load_offset()

    def _load_offset(self) -> int:
        try:
            return int(self._offset_file.read_text().strip())
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        try:
            self._offset_file.write_text(str(offset))
        except Exception:
            pass

    def iter_new(self) -> Iterator[dict]:
        """Yield all events appended since last call. Updates offset on return."""
        if not _BUS_FILE.exists():
            return
        try:
            with _BUS_FILE.open("r", encoding="utf-8") as f:
                f.seek(self._offset)
                new_events = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        new_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                new_offset = f.tell()

            for event in new_events:
                yield event

            if new_offset != self._offset:
                self._offset = new_offset
                self._save_offset(new_offset)

        except Exception as exc:
            logger.debug("[EventBus] read failed for %s: %s", self._name, exc)

    def peek_recent(self, n: int = 10, event_type: str | None = None) -> list[dict]:
        """Return last N events (optionally filtered by type) without updating offset."""
        if not _BUS_FILE.exists():
            return []
        try:
            lines = _BUS_FILE.read_text(encoding="utf-8").splitlines()
            events = []
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if event_type is None or e.get("type") == event_type:
                        events.append(e)
                        if len(events) >= n:
                            break
                except json.JSONDecodeError:
                    pass
            return list(reversed(events))
        except Exception:
            return []
