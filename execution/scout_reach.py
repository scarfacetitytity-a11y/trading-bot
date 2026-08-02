"""AiDEN Scout Reach — autonomous internet intelligence layer.

Adapted from Agent-Reach (github.com/Panniantong/Agent-Reach):
  - Channel abstraction: each source = a Channel with ordered backends + probe
  - Jina Web Reader: https://r.jina.ai/{url} → clean Markdown, zero-config
  - RSS via feedparser: Reuters, DailyFX economic headlines
  - ForexFactory JSON API (primary) → Jina calendar fallback

Zero tokens. Runs as a standalone process. Writes pre-session briefs to
Obsidian Brain/Scout/ and sends a Telegram morning summary at RUN_HOUR UTC.

Run: python -m execution.scout_reach
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT      = Path(__file__).parent.parent
LOG_DIR   = ROOT / "logs"
PID_FILE  = LOG_DIR / "scout_reach.pid"
STATE_FILE= LOG_DIR / "scout_reach_state.json"
VAULT     = Path(r"C:\Users\anton\OneDrive\Desktop\Aiden\AiDEN\Brain\Scout")
ENV_FILE  = ROOT / ".env"

RUN_HOUR  = 6    # UTC hour to generate daily brief
POLL_SECS = 60   # check-cycle interval

# cp1252 console chokes on →/— in log messages
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | scout | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scout_reach.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; AiDEN-Scout/1.0)"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def _tg(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    try:
        payload = json.dumps({
            "chat_id": chat, "text": f"[AiDEN Scout] {text}", "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.debug("Telegram failed: %s", exc)

def _get(url: str, timeout: int = 15) -> str:
    """Simple HTTP GET → UTF-8 string."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json, text/plain, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

# ── Channel abstraction (adapted from Agent-Reach) ────────────────────────────

@dataclass
class ChannelResult:
    status: str        # "ok" | "warn" | "off" | "error"
    data:   str = ""
    source: str = ""   # which backend served this


class Channel(ABC):
    name:     str = ""
    backends: list[str] = field(default_factory=list)

    @abstractmethod
    def fetch(self) -> ChannelResult:
        ...

    def probe(self) -> str:
        """Return 'ok' or reason why channel is unavailable."""
        try:
            r = self.fetch()
            return r.status
        except Exception as exc:
            return f"error: {exc}"


# ── Channel 1: ForexFactory Economic Calendar ─────────────────────────────────

HIGH_IMPACT_EVENTS = {
    "nfp", "non-farm", "payroll", "unemployment", "cpi", "inflation",
    "fomc", "fed", "interest rate", "gdp", "retail sales",
    "pmi", "ism", "ppe", "boe", "ecb", "boj", "rba", "rbnz",
    "jolts", "adp", "pce",
}

class ForexCalendarChannel(Channel):
    name = "forex_calendar"
    backends = ["ForexFactory JSON", "Jina Web Reader"]

    def fetch(self) -> ChannelResult:
        # Backend 0: FF JSON API
        try:
            raw = _get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            events = json.loads(raw)
            today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            high   = [
                e for e in events
                if e.get("impact", "").lower() in ("high", "red")
                and today in (e.get("date", "") or "")
            ]
            return ChannelResult("ok", data=json.dumps(high), source="ForexFactory JSON")
        except Exception as exc:
            logger.warning("FF JSON failed: %s — trying Jina fallback", exc)

        # Backend 1: Jina reader on investing.com economic calendar
        try:
            page = _get("https://r.jina.ai/https://www.investing.com/economic-calendar/", timeout=20)
            # Extract lines containing currency + impact words
            lines = [l.strip() for l in page.splitlines()
                     if any(w in l.lower() for w in HIGH_IMPACT_EVENTS)]
            summary = "\n".join(lines[:20]) if lines else "No high-impact events found."
            return ChannelResult("ok", data=summary, source="Jina/Investing.com")
        except Exception as exc2:
            return ChannelResult("error", data=str(exc2), source="all backends failed")


# ── Channel 2: RSS Headline Feed ──────────────────────────────────────────────

RSS_FEEDS = [
    ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    ("DailyFX Forex",     "https://www.dailyfx.com/feeds/forex-market-news"),
    ("ForexLive",         "https://www.forexlive.com/feed/news"),
]

FX_KEYWORDS = {
    "fed", "fomc", "dollar", "usd", "gbp", "eur", "jpy", "gold", "xau",
    "inflation", "cpi", "nfp", "rate", "boe", "ecb", "boj",
    "risk", "recession", "yield", "treasury", "dxy",
}

class RSSChannel(Channel):
    name = "rss"
    backends = ["feedparser", "Jina Reader"]

    def _feedparser_available(self) -> bool:
        try:
            import feedparser  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self) -> ChannelResult:
        headlines: list[str] = []
        used_backend = "none"

        if self._feedparser_available():
            import feedparser
            for feed_name, url in RSS_FEEDS:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:8]:
                        title = entry.get("title", "")
                        if any(kw in title.lower() for kw in FX_KEYWORDS):
                            headlines.append(f"[{feed_name}] {title}")
                    used_backend = "feedparser"
                except Exception as exc:
                    logger.debug("RSS %s failed: %s", feed_name, exc)

        # Jina fallback if feedparser unavailable or no headlines found
        if not headlines:
            try:
                page = _get("https://r.jina.ai/https://www.forexlive.com", timeout=20)
                lines = [l.strip() for l in page.splitlines()
                         if any(kw in l.lower() for kw in FX_KEYWORDS) and len(l) > 30]
                headlines = lines[:10]
                used_backend = "Jina/ForexLive"
            except Exception as exc:
                logger.debug("Jina ForexLive fallback failed: %s", exc)

        if not headlines:
            return ChannelResult("warn", data="No FX headlines found.", source=used_backend)
        return ChannelResult("ok", data="\n".join(headlines[:10]), source=used_backend)


# ── Channel 3: DXY Context ────────────────────────────────────────────────────

class DXYContextChannel(Channel):
    name = "dxy_context"
    backends = ["Jina/ForexLive-DXY"]

    def fetch(self) -> ChannelResult:
        try:
            page = _get("https://r.jina.ai/https://www.forexlive.com/tag/dxy/", timeout=20)
            lines = [l.strip() for l in page.splitlines()
                     if any(w in l.lower() for w in ("dollar", "dxy", "index", "strength", "weak"))
                     and 20 < len(l) < 300]
            snippet = "\n".join(lines[:6]) if lines else "DXY context unavailable."
            return ChannelResult("ok", data=snippet, source="Jina/ForexLive")
        except Exception as exc:
            return ChannelResult("error", data=str(exc), source="Jina/ForexLive")


# ── Brief builder ─────────────────────────────────────────────────────────────

def _parse_ff_events(data: str) -> list[dict]:
    try:
        return json.loads(data)
    except Exception:
        return []


def build_brief(date_str: str) -> str:
    """Fetch all channels and assemble the pre-session brief."""
    cal_ch  = ForexCalendarChannel()
    rss_ch  = RSSChannel()
    dxy_ch  = DXYContextChannel()

    logger.info("Fetching economic calendar...")
    cal_r = cal_ch.fetch()

    logger.info("Fetching RSS headlines...")
    rss_r = rss_ch.fetch()

    logger.info("Fetching DXY context...")
    dxy_r = dxy_ch.fetch()

    lines = [
        f"---",
        f"type: brief",
        f"status: active",
        f"tags: [scout, brief, daily]",
        f"relatedTo: [AiDEN, Trading]",
        f"date: {date_str}",
        f"---",
        f"",
        f"# AiDEN Scout — {date_str}",
        f"",
        f"## High-Impact Events Today",
    ]

    if cal_r.status == "ok":
        events = _parse_ff_events(cal_r.data)
        if events:
            for ev in events[:8]:
                time_str = ev.get("date", "")[-5:] if "T" in ev.get("date","") else ev.get("date","")
                lines.append(f"- **{ev.get('title', '?')}** ({ev.get('currency', '?')}) @ {time_str} UTC — impact: {ev.get('impact','?')}")
        elif isinstance(cal_r.data, str) and not cal_r.data.startswith("["):
            # Jina fallback text
            lines += [f"  {l}" for l in cal_r.data.splitlines()[:10]]
        else:
            lines.append("- No high-impact events today.")
    else:
        lines.append(f"- Calendar unavailable ({cal_r.status}): {cal_r.data[:80]}")

    lines += ["", f"*Source: {cal_r.source}*", "", "## FX Headlines"]

    if rss_r.status in ("ok", "warn"):
        for hl in rss_r.data.splitlines()[:10]:
            lines.append(f"- {hl}")
    else:
        lines.append(f"- Headlines unavailable: {rss_r.data[:80]}")

    lines += ["", f"*Source: {rss_r.source}*", "", "## DXY Context"]

    if dxy_r.status == "ok":
        lines += [f"  {l}" for l in dxy_r.data.splitlines()[:6]]
    else:
        lines.append("- DXY context unavailable.")

    lines += ["", f"*Source: {dxy_r.source}*", "", "## Links", "[[AiDEN]] | [[Trading]]"]

    return "\n".join(lines)


def _write_obsidian(date_str: str, content: str) -> None:
    VAULT.mkdir(parents=True, exist_ok=True)
    out = VAULT / f"{date_str}-scout-brief.md"
    out.write_text(content, encoding="utf-8")
    logger.info("Brief written to %s", out)


def _brief_telegram(events: list[dict], headlines: list[str]) -> str:
    parts = ["<b>AiDEN Scout Morning Brief</b>"]
    if events:
        parts.append("\n<b>High-Impact Today:</b>")
        for ev in events[:5]:
            t = ev.get("date", "")[-5:] if "T" in ev.get("date","") else ""
            parts.append(f"  • {ev.get('title','?')} ({ev.get('currency','?')}) {t} UTC")
    else:
        parts.append("\nNo high-impact events today.")
    if headlines:
        parts.append("\n<b>Headlines:</b>")
        for h in headlines[:4]:
            # Strip RSS prefix
            h = re.sub(r"^\[.*?\]\s*", "", h)
            parts.append(f"  • {h[:80]}")
    return "\n".join(parts)


# ── State ─────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_daily_brief() -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Running daily brief for %s", date_str)
    try:
        content = build_brief(date_str)
        _write_obsidian(date_str, content)

        # Extract for Telegram
        events = []
        try:
            cal_data = _get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            events = [e for e in json.loads(cal_data)
                      if e.get("impact", "").lower() in ("high", "red") and today in (e.get("date") or "")]
        except Exception:
            pass

        headlines = []
        rss_r = RSSChannel().fetch()
        if rss_r.status in ("ok", "warn"):
            headlines = rss_r.data.splitlines()

        _tg(_brief_telegram(events, headlines))
        logger.info("Daily brief complete.")
        return date_str
    except Exception as exc:
        logger.error("Brief failed: %s", exc)
        _tg(f"<b>Scout brief failed:</b> {exc}")
        return date_str


def main() -> None:
    _load_env()
    LOG_DIR.mkdir(exist_ok=True)

    my_pid = os.getpid()
    existing = None
    try:
        existing = int(PID_FILE.read_text().strip())
    except Exception:
        pass
    if existing and existing != my_pid:
        try:
            os.kill(existing, 0)
            logger.error("Scout already running (PID %d) — exiting", existing)
            sys.exit(1)
        except (OSError, ProcessLookupError):
            pass
    PID_FILE.write_text(str(my_pid))

    logger.info("Scout Reach started (PID %d)", my_pid)

    state = _load_state()
    try:
        while True:
            now   = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            # Run brief once per day at/after RUN_HOUR UTC — >= so a late
            # process start still produces the brief (catch-up semantics)
            if now.hour >= RUN_HOUR and state.get("last_brief") != today:
                ran_date = run_daily_brief()
                state["last_brief"] = ran_date
                _save_state(state)
            time.sleep(POLL_SECS)
    except KeyboardInterrupt:
        logger.info("Scout stopped.")
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
