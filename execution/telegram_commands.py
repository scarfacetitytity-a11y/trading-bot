"""Two-way Telegram command handler for AiDEN bot.

Runs a background polling thread listening for messages in the bot's chat.
Supported commands (prefix / or without prefix):

  /status   — Dashboard snapshot: equity, DD, open positions, active signals
  /pause    — Set soft halt: no new entries until /resume
  /resume   — Clear soft halt: re-enable new entries
  /flatten  — Close all open positions immediately
  /why      — Explain the last skipped signal (which gate blocked it)
  /risk N   — Set risk_pct to N (e.g. /risk 0.75)

Security: only responds to messages from the configured TELEGRAM_CHAT_ID.
All commands are logged. Destructive commands (/flatten) require confirmation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _load_credentials() -> tuple[str, str]:
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        os.environ.get("TELEGRAM_CHAT_ID", ""),
    )


_TOKEN, _CHAT_ID = _load_credentials()


def _tg_send(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        url     = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": _CHAT_ID,
            "text":    text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8)
    except Exception as exc:
        logger.warning("[TgCmd] send failed: %s", exc)


def _tg_get_updates(offset: int, timeout: int = 10) -> list:
    if not _TOKEN:
        return []
    try:
        url  = (
            f"https://api.telegram.org/bot{_TOKEN}/getUpdates"
            f"?offset={offset}&timeout={timeout}&allowed_updates=message"
        )
        req  = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout + 5)
        data = json.loads(resp.read())
        return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


class TelegramCommandHandler:
    """Background command listener for two-way Telegram control.

    Inject callbacks from the orchestrator:
      status_fn:  () → str         dashboard text
      pause_fn:   () → None        set soft halt
      resume_fn:  () → None        clear soft halt
      flatten_fn: () → str         close all positions, return confirmation
      why_fn:     () → str         last gate that blocked entry + context
      set_risk_fn:(float) → str    change risk_pct, return new value string
    """

    POLL_INTERVAL = 1   # seconds between getUpdates polls (long-poll handles idle)
    LONG_POLL_TO  = 10  # seconds for Telegram long-poll timeout

    def __init__(
        self,
        status_fn:   Optional[Callable[[], str]]    = None,
        pause_fn:    Optional[Callable[[], None]]   = None,
        resume_fn:   Optional[Callable[[], None]]   = None,
        flatten_fn:  Optional[Callable[[], str]]    = None,
        why_fn:      Optional[Callable[[], str]]    = None,
        set_risk_fn: Optional[Callable[[float], str]] = None,
    ) -> None:
        self._status_fn   = status_fn
        self._pause_fn    = pause_fn
        self._resume_fn   = resume_fn
        self._flatten_fn  = flatten_fn
        self._why_fn      = why_fn
        self._set_risk_fn = set_risk_fn

        self._offset      = 0
        self._stop        = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_flatten = False   # awaiting /flatten confirm

    def start(self) -> None:
        if not _TOKEN or not _CHAT_ID:
            logger.warning("[TgCmd] No credentials — command handler disabled")
            return
        self._thread = threading.Thread(
            target=self._poll_loop, name="TgCommandHandler", daemon=True
        )
        self._thread.start()
        logger.info("[TgCmd] Started — listening for commands in chat %s", _CHAT_ID)

    def stop(self) -> None:
        self._stop.set()

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            updates = _tg_get_updates(self._offset, self.LONG_POLL_TO)
            for upd in updates:
                self._offset = max(self._offset, upd.get("update_id", 0) + 1)
                try:
                    self._handle_update(upd)
                except Exception as exc:
                    logger.warning("[TgCmd] handle error: %s", exc)
            # No sleep needed — long-poll already holds for LONG_POLL_TO seconds

    def _handle_update(self, upd: dict) -> None:
        msg = upd.get("message", {})
        if not msg:
            return

        # Security: only respond to the authorised chat
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != str(_CHAT_ID):
            logger.warning("[TgCmd] message from unknown chat %s — ignored", chat_id)
            return

        text = (msg.get("text") or "").strip().lower()
        if not text.startswith("/"):
            text = "/" + text

        logger.info("[TgCmd] received: %s", text)

        if text in ("/status", "/s"):
            self._cmd_status()
        elif text in ("/pause", "/p"):
            self._cmd_pause()
        elif text in ("/resume", "/r"):
            self._cmd_resume()
        elif text in ("/flatten", "/f"):
            self._cmd_flatten_confirm()
        elif text in ("/flatten yes", "/flatten confirm", "/f yes"):
            self._cmd_flatten_execute()
        elif text in ("/why", "/w"):
            self._cmd_why()
        elif text.startswith("/risk "):
            parts = text.split()
            if len(parts) == 2:
                try:
                    self._cmd_set_risk(float(parts[1]))
                except ValueError:
                    _tg_send("Usage: /risk 0.75")
        elif text in ("/help", "/h", "/?"):
            self._cmd_help()
        else:
            _tg_send(f"Unknown command: <code>{text}</code>\nSend /help for commands.")

    # ── Command implementations ───────────────────────────────────────────────

    def _cmd_status(self) -> None:
        if self._status_fn:
            try:
                reply = self._status_fn()
            except Exception as exc:
                reply = f"Status error: {exc}"
        else:
            reply = "Status callback not wired."
        _tg_send(f"<b>AiDEN Status</b> {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n\n{reply}")

    def _cmd_pause(self) -> None:
        if self._pause_fn:
            try:
                self._pause_fn()
                _tg_send("<b>PAUSED</b> — soft halt set. No new entries until /resume.")
            except Exception as exc:
                _tg_send(f"Pause error: {exc}")
        else:
            _tg_send("Pause callback not wired.")

    def _cmd_resume(self) -> None:
        if self._resume_fn:
            try:
                self._resume_fn()
                _tg_send("<b>RESUMED</b> — soft halt cleared. New entries allowed.")
            except Exception as exc:
                _tg_send(f"Resume error: {exc}")
        else:
            _tg_send("Resume callback not wired.")

    def _cmd_flatten_confirm(self) -> None:
        self._pending_flatten = True
        _tg_send(
            "⚠️ <b>FLATTEN ALL POSITIONS</b>\n"
            "This will close every open trade immediately.\n\n"
            "Reply <code>/flatten yes</code> to confirm, or anything else to cancel."
        )

    def _cmd_flatten_execute(self) -> None:
        if not self._pending_flatten:
            _tg_send("No flatten pending. Send /flatten first.")
            return
        self._pending_flatten = False
        if self._flatten_fn:
            try:
                result = self._flatten_fn()
                _tg_send(f"<b>FLATTEN EXECUTED</b>\n{result}")
            except Exception as exc:
                _tg_send(f"Flatten error: {exc}")
        else:
            _tg_send("Flatten callback not wired.")

    def _cmd_why(self) -> None:
        if self._why_fn:
            try:
                reply = self._why_fn()
            except Exception as exc:
                reply = f"Why error: {exc}"
        else:
            reply = "Why callback not wired — check logs/shadow_stack.jsonl for last entry."
        _tg_send(f"<b>Last blocked signal</b>\n\n{reply}")

    def _cmd_set_risk(self, pct: float) -> None:
        if not (0.1 <= pct <= 5.0):
            _tg_send("Risk pct must be between 0.1 and 5.0.")
            return
        if self._set_risk_fn:
            try:
                reply = self._set_risk_fn(pct)
                _tg_send(f"Risk updated: <code>{reply}</code>")
            except Exception as exc:
                _tg_send(f"Set risk error: {exc}")
        else:
            _tg_send("Set-risk callback not wired.")

    def _cmd_help(self) -> None:
        _tg_send(
            "<b>AiDEN Commands</b>\n\n"
            "/status (s)   — dashboard snapshot\n"
            "/pause  (p)   — halt new entries\n"
            "/resume (r)   — re-enable entries\n"
            "/flatten (f)  — close all positions (requires confirm)\n"
            "/why (w)      — last skipped signal explanation\n"
            "/risk N       — set risk_pct (e.g. /risk 0.75)\n"
            "/help         — this message"
        )
