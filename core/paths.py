"""Portable path resolution for machine-specific locations.

Every location that used to be hard-coded to one Windows profile
(C:\\Users\\anton\\...) resolves here, in this order:

    1. environment variable (AIDEN_*)
    2. config/config.yaml value, where one exists
    3. the legacy Windows default — only on Windows, so the original machine
       keeps working with zero configuration
    4. a home-relative default on every other OS

Nothing in this module reads or exposes secrets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

BOT_ROOT = Path(__file__).resolve().parent.parent

_IS_WINDOWS = os.name == "nt"

_LEGACY_VAULT        = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden")
_LEGACY_FIRM_DB      = Path(r"C:\Users\anton\.firm\firm.db")
_LEGACY_FINANCIAL_MCP = Path(r"C:\Users\anton\Documents\financial-mcp")


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else None


def _config_value(section: str, key: str) -> Optional[str]:
    try:
        from config.settings import load_config
        return (load_config().get(section, {}) or {}).get(key) or None
    except Exception:
        return None


def logs_dir() -> Path:
    return _env_path("AIDEN_LOGS_DIR") or (BOT_ROOT / "logs")


def vault_root() -> Path:
    """Obsidian vault root (the folder that contains AiDEN/)."""
    p = _env_path("AIDEN_VAULT_PATH")
    if p:
        return p
    cv = _config_value("obsidian", "vault_path")
    if cv:
        return Path(cv).expanduser()
    return _LEGACY_VAULT if _IS_WINDOWS else Path.home() / "aiden-vault"


def aiden_dir() -> Path:
    return vault_root() / "AiDEN"


def brain_dir() -> Path:
    return aiden_dir() / "Brain"


def firm_db() -> Path:
    p = _env_path("AIDEN_FIRM_DB")
    if p:
        return p
    return _LEGACY_FIRM_DB if _IS_WINDOWS else Path.home() / ".firm" / "firm.db"


def financial_mcp_dir() -> Path:
    p = _env_path("AIDEN_FINANCIAL_MCP_DIR")
    if p:
        return p
    return _LEGACY_FINANCIAL_MCP if _IS_WINDOWS else Path.home() / "financial-mcp"


def claude_cli() -> str:
    """Claude Code CLI executable used for headless Cadre invocations."""
    explicit = os.environ.get("AIDEN_CLAUDE_CLI", "").strip()
    if explicit:
        return explicit
    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "claude.cmd")
    if _IS_WINDOWS:
        candidates.append(Path(r"C:\Program Files\nodejs\claude.cmd"))
    for c in candidates:
        if c.exists():
            return str(c)
    return "claude"
