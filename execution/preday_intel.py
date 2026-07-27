"""Pre-session global intelligence briefing.

Runs at session start (configurable, default 06:30 UTC) and produces a
structured brief covering:
  1. Economic calendar — high-impact events today and tomorrow
  2. Macro news — top 5 market-moving headlines
  3. DXY context — current bias and what it means for each instrument
  4. Geopolitical / risk-off signals
  5. Instrument-level watchlist — which are cleanest for today

Writes to: AiDEN/System/intel_brief.md (vault) and sends Telegram summary.
Uses financial-mcp tools via direct import (server must be running) or
falls back to direct API calls.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRIEF_INSTRUMENTS = [
    "XAUUSD", "GBPUSD", "EURUSD", "USDJPY",
    "US30", "US100", "US500", "JP225",
]

DXY_IMPACTS = {
    "XAUUSD": ("inverse", "DXY up = gold headwind"),
    "XAGUSD": ("inverse", "DXY up = silver headwind"),
    "GBPUSD": ("inverse", "DXY up = cable drops"),
    "EURUSD": ("inverse", "DXY up = euro drops (57% of DXY)"),
    "USDJPY": ("direct",  "DXY up = USD/JPY rises (yen weakens)"),
    "US30":   ("loose_inverse", "DXY up can pressure equities"),
    "US100":  ("loose_inverse", "DXY up can pressure tech"),
    "US500":  ("loose_inverse", "DXY up can pressure S&P"),
    "JP225":  ("yen_driven", "Yen weakens → Nikkei rises (export boost)"),
}


def _call_financial_mcp(tool: str, args: dict) -> Optional[dict]:
    """Call financial-mcp server tool via subprocess JSON-RPC."""
    try:
        payload = json.dumps({"tool": tool, "args": args})
        result = subprocess.run(
            ["python", "-m", "financial_mcp.client", payload],
            capture_output=True, text=True, timeout=15,
            cwd=Path(r"C:\Users\anton\Documents\financial-mcp"),
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception as e:
        logger.debug("[Intel] MCP call failed (%s): %s", tool, e)
    return None


def _fetch_calendar() -> list[dict]:
    """Get today's high-impact economic events."""
    try:
        import sys; sys.path.insert(0, r"C:\Users\anton\Documents\financial-mcp")
        from tools import get_economic_calendar
        data = get_economic_calendar()
        if isinstance(data, list):
            return [e for e in data if e.get("impact", "").lower() in ("high", "red")]
        if isinstance(data, str):
            return [{"raw": data}]
    except Exception as e:
        logger.debug("[Intel] Calendar fetch failed: %s", e)
    return []


def _fetch_news() -> list[str]:
    """Get top market-moving headlines."""
    try:
        import sys; sys.path.insert(0, r"C:\Users\anton\Documents\financial-mcp")
        from tools import get_financial_news
        data = get_financial_news(query="forex gold indices market", limit=5)
        if isinstance(data, list):
            return [f"• {item.get('title', item)}" for item in data[:5]]
        if isinstance(data, str):
            lines = [l.strip() for l in data.split("\n") if l.strip()]
            return [f"• {l}" for l in lines[:5]]
    except Exception as e:
        logger.debug("[Intel] News fetch failed: %s", e)
    return []


def _fetch_dxy_price() -> Optional[float]:
    """Get current DXY level."""
    try:
        import sys; sys.path.insert(0, r"C:\Users\anton\Documents\financial-mcp")
        from tools import get_forex_price
        data = get_forex_price("DXY")
        if isinstance(data, dict):
            return data.get("price") or data.get("close")
        if isinstance(data, (int, float)):
            return float(data)
    except Exception as e:
        logger.debug("[Intel] DXY fetch failed: %s", e)
    return None


def generate_brief() -> str:
    """Generate the full pre-session intelligence brief as markdown."""
    now = datetime.now(tz=timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []
    sections.append(f"# AiDEN — Pre-Session Intelligence Brief\n**{date_str}**\n")

    # ── Economic Calendar ─────────────────────────────────────────────────────
    calendar = _fetch_calendar()
    sections.append("## High-Impact Events Today")
    if calendar:
        for event in calendar[:8]:
            if "raw" in event:
                sections.append(event["raw"])
            else:
                time  = event.get("time", event.get("date", "?"))
                name  = event.get("event", event.get("name", "?"))
                curr  = event.get("currency", event.get("country", ""))
                sections.append(f"- **{time}** `{curr}` {name}")
    else:
        sections.append("- No high-impact events fetched (check financial-mcp)")

    # ── Market News ───────────────────────────────────────────────────────────
    news = _fetch_news()
    sections.append("\n## Top Headlines")
    if news:
        sections.extend(news)
    else:
        sections.append("- No headlines fetched")

    # ── DXY Context ───────────────────────────────────────────────────────────
    dxy = _fetch_dxy_price()
    sections.append("\n## DXY — Dollar Index")
    if dxy:
        sections.append(f"**Current level: {dxy:.3f}**\n")
    sections.append("| Instrument | DXY Relationship | Implication |")
    sections.append("|---|---|---|")
    for sym, (rel, impl) in DXY_IMPACTS.items():
        sections.append(f"| {sym} | {rel} | {impl} |")

    # ── Risk Radar ────────────────────────────────────────────────────────────
    sections.append("\n## Risk Radar")
    sections.append("Watch these before entering:")
    sections.append("- USDJPY rapid drop → yen carry unwind → global risk-off → reduce all longs")
    sections.append("- US10Y yield spike → equity pressure → indices shorts preferred")
    sections.append("- VIX > 20 → reduce size on all equity instruments by 50%")
    sections.append("- Fed speaker scheduled → widen stops or stand aside 30 min either side")

    # ── Session Windows ───────────────────────────────────────────────────────
    sections.append("\n## Today's Session Windows (UTC)")
    sections.append("| Session | Time | Best instruments |")
    sections.append("|---|---|---|")
    sections.append("| Tokyo | 00-09 | JP225, USDJPY |")
    sections.append("| London | 07-12 | GBPUSD, EURUSD, XAUUSD |")
    sections.append("| **Crossover** | **12-17** | **All — highest volume** |")
    sections.append("| New York | 12-21 | US30, US100, US500, XAUUSD, Silver |")

    sections.append("\n---\n*Auto-generated by AiDEN pre-session engine*")
    return "\n".join(sections)


def write_brief_to_vault(vault_path: Optional[Path] = None) -> Path:
    """Write the brief to the Obsidian vault."""
    if vault_path is None:
        try:
            from config.settings import load_config
            cfg = load_config()
            vp = cfg.get("obsidian", {}).get("vault_path")
            vault_path = Path(vp) if vp else Path(r"C:\Users\anton\OneDrive\Desktop\Aiden")
        except Exception:
            vault_path = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden")

    out = vault_path / "AiDEN" / "System" / "intel_brief.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    content = generate_brief()
    out.write_text(content, encoding="utf-8")
    logger.info("[Intel] Brief written to %s", out)
    return out


def send_telegram_brief(token: str, chat_id: str) -> None:
    """Send a condensed brief to Telegram."""
    try:
        import requests
        brief = generate_brief()
        # Telegram has 4096 char limit — send first section only
        lines = brief.split("\n")
        msg_lines = []
        total = 0
        for line in lines:
            if total + len(line) > 3800:
                break
            msg_lines.append(line)
            total += len(line)
        msg = "\n".join(msg_lines)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        logger.info("[Intel] Brief sent to Telegram")
    except Exception as e:
        logger.warning("[Intel] Telegram send failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = write_brief_to_vault()
    print(f"Brief written to: {path}")
    print(generate_brief())
