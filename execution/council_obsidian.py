"""Obsidian writer utility — shared by council_watch and hooks.

Writes structured Brain notes and session updates to the AiDEN Obsidian vault
with proper MOP frontmatter so base sync can extract them into the graph.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VAULT_ROOT   = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden")
BRAIN_DIR    = VAULT_ROOT / "AiDEN" / "Brain"
SESSIONS_DIR = VAULT_ROOT / "AiDEN" / "Sessions"
TRADES_DIR   = VAULT_ROOT / "AiDEN" / "Trades"
ALERTS_DIR   = VAULT_ROOT / "AiDEN" / "Alerts"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def write_brain_note(
    slug: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
    related: list[str] | None = None,
    note_type: str = "insight",
) -> Path:
    """Write a Brain note with MOP frontmatter. Returns the file path."""
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    tags    = tags or ["aiden", "council"]
    related = related or ["AiDEN", "Brain"]

    tags_yaml    = "[" + ", ".join(tags) + "]"
    related_yaml = "[" + ", ".join(related) + "]"

    content = f"""---
type: {note_type}
status: active
tags: {tags_yaml}
relatedTo: {related_yaml}
date: {_today()}
---

# {title}

{body}

---
_Written by council_watch at {_utcnow()}_
"""
    path = BRAIN_DIR / f"{_today()}-{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_alert(
    alert_id: str,
    title: str,
    severity: str,
    body: str,
    tags: list[str] | None = None,
) -> Path:
    """Write a time-stamped alert note to Alerts/. severity: critical|warning|info"""
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    tags = tags or ["alert", "aiden", severity]
    tags_yaml = "[" + ", ".join(tags) + "]"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    content = f"""---
type: alert
status: active
tags: {tags_yaml}
relatedTo: [AiDEN, Trading]
date: {_today()}
severity: {severity}
---

# [{severity.upper()}] {title}

{body}

---
_Alert {alert_id} — {_utcnow()}_
"""
    path = ALERTS_DIR / f"{ts}-{alert_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


def update_session_note(key: str, value: Any) -> None:
    """Upsert a key-value row into today's session note table section."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    today     = _today()
    note_path = SESSIONS_DIR / f"{today}.md"

    if not note_path.exists():
        note_path.write_text(f"""---
type: session
status: active
tags: [session, daily, aiden]
relatedTo: [AiDEN, Trading]
date: {today}
---

# Session {today}

## Council Watch
| Key | Value | Time |
|---|---|---|

## Links
[[AiDEN]] | [[Trading]] | [[Brain/]]
""", encoding="utf-8")

    text = note_path.read_text(encoding="utf-8")
    row  = f"| {key} | {value} | {datetime.now(timezone.utc).strftime('%H:%M UTC')} |"

    if "## Council Watch" not in text:
        text += f"\n## Council Watch\n| Key | Value | Time |\n|---|---|---|\n{row}\n"
    else:
        # Append row after the table header
        lines = text.splitlines()
        insert_after = -1
        in_table = False
        for i, line in enumerate(lines):
            if "## Council Watch" in line:
                in_table = True
            if in_table and line.startswith("|---|"):
                insert_after = i
            if in_table and insert_after > 0 and not line.startswith("|") and i > insert_after + 1:
                break

        if insert_after >= 0:
            lines.insert(insert_after + 1, row)
            text = "\n".join(lines)

    note_path.write_text(text, encoding="utf-8")


def append_to_session(section_title: str, content: str) -> None:
    """Append a freeform section to today's session note."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    today     = _today()
    note_path = SESSIONS_DIR / f"{today}.md"

    block = f"\n## {section_title}\n{content}\n"

    if note_path.exists():
        existing = note_path.read_text(encoding="utf-8")
        note_path.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")
    else:
        note_path.write_text(f"""---
type: session
status: active
tags: [session, daily, aiden]
relatedTo: [AiDEN, Trading]
date: {today}
---

# Session {today}
{block}
## Links
[[AiDEN]] | [[Trading]] | [[Brain/]]
""", encoding="utf-8")
