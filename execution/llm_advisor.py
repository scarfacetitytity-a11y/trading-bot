"""Multi-model LLM advisor for AiDEN governance advisory layer.

Calls 2-3 free/cheap LLM APIs with the same prompt and returns their
responses as additional (non-authoritative) notes for the CouncilRouter.

Design constraints:
- NEVER blocks the main trading loop — all model calls run concurrently
  under a single TOTAL_TIMEOUT_S deadline via concurrent.futures.
- NEVER changes trade approval — advisory text only, appended to notes.
- Degrades silently to {} if keys are missing or requests is unavailable.
- Models are OpenAI-compatible /chat/completions endpoints (OpenRouter, Groq).

Credentials (environment variables, set in .env):
  OPENROUTER_API_KEY — https://openrouter.ai/   (free tier: 20 req/min)
  GROQ_API_KEY       — https://console.groq.com/ (free tier: 14400 req/day)
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any

logger = logging.getLogger(__name__)

# ── Model definitions ──────────────────────────────────────────────────────────
# Update these when free model IDs change. Bad IDs yield an error string, not a crash.

_OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
_GROQ_BASE       = "https://api.groq.com/openai/v1/chat/completions"

# Free/zero-cost model IDs as of mid-2026 — adjust as providers update
_DEFAULT_MODELS: list[dict] = [
    {
        "id":       "openrouter/qwen/qwen3-8b:free",
        "endpoint": _OPENROUTER_BASE,
        "key_env":  "OPENROUTER_API_KEY",
        "label":    "OpenRouter/Qwen3-8B",
    },
    {
        "id":       "llama-3.1-8b-instant",
        "endpoint": _GROQ_BASE,
        "key_env":  "GROQ_API_KEY",
        "label":    "Groq/Llama3.1-8B",
    },
]

# Single bounded wall-clock ceiling for ALL parallel model calls combined
TOTAL_TIMEOUT_S = 10


# ── Internals ──────────────────────────────────────────────────────────────────

def _have_requests() -> bool:
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def _call_model(model: dict, prompt: str) -> tuple[str, str]:
    """Single synchronous model call. Returns (label, response_text_or_error)."""
    import requests  # guarded — only called after _have_requests() check

    api_key = os.environ.get(model["key_env"], "")
    if not api_key:
        return model["label"], "[key not set]"

    payload = {
        "model": model["id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(
            model["endpoint"],
            json=payload,
            headers=headers,
            timeout=9,  # per-request slightly under the global ceiling
        )
        if resp.status_code != 200:
            return model["label"], f"[HTTP {resp.status_code}]"
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return model["label"], text or "[empty]"
    except Exception as exc:
        return model["label"], f"[error: {exc}]"


# ── Public API ─────────────────────────────────────────────────────────────────

def query(prompt: str, models: list[dict] | None = None) -> dict[str, str]:
    """Call each model concurrently, return {label: response}.

    Returns {} immediately if requests is not installed or no keys are set.
    All calls are bounded by TOTAL_TIMEOUT_S — slower models are abandoned.
    """
    if not _have_requests():
        return {}

    target_models = models if models is not None else _DEFAULT_MODELS

    # Quick-exit: skip if no keys at all (avoids spinning up executor for nothing)
    configured = [m for m in target_models if os.environ.get(m["key_env"], "")]
    if not configured:
        return {}

    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(configured)) as pool:
        futures = {pool.submit(_call_model, m, prompt): m["label"] for m in configured}
        try:
            for fut in as_completed(futures, timeout=TOTAL_TIMEOUT_S):
                label, text = fut.result()
                results[label] = text
        except FuturesTimeout:
            # Record which models didn't respond in time
            for fut, label in futures.items():
                if label not in results and not fut.done():
                    results[label] = "[timeout]"
                elif label not in results and fut.done():
                    try:
                        label2, text2 = fut.result()
                        results[label2] = text2
                    except Exception:
                        results[label] = "[error]"

    return results


def trade_advisory(
    symbol: str,
    direction: int,
    score: int,
    daily_dd_pct: float,
) -> dict[str, str]:
    """Format a structured trade advisory prompt and query all configured models.

    Returns {model_label: advisory_text}. Returns {} if no keys configured.

    The prompt is intentionally terse — we want a single sentence of perspective,
    not an essay. Models with 200 max_tokens won't hallucinate multi-page fiction.
    """
    dir_str = "LONG" if direction > 0 else "SHORT"
    prompt = (
        f"AiDEN FTMO $100k trading bot — pending trade advisory:\n"
        f"Symbol: {symbol} | Direction: {dir_str} | Score: {score}/10 | "
        f"Daily drawdown so far: {daily_dd_pct:.2f}%\n\n"
        f"As a senior trading risk advisor, give ONE sentence of the most important "
        f"risk consideration for this trade. Be direct. No preamble."
    )
    return query(prompt)
