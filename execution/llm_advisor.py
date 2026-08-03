"""Multi-model LLM advisor — AiDEN's AI model hub.

Three layers:
  1. Trading advisory  — CouncilRouter calls trade_advisory() per trade entry
  2. General query     — query(prompt) returns {model_label: text} from all live models
  3. CLI interface     — `python -m execution.llm_advisor "your question"` for terminal use

Supported providers (all free tiers, OpenAI-compatible /chat/completions):
  OPENROUTER_API_KEY  — openrouter.ai      20 req/min free; routes to 100+ models
  GROQ_API_KEY        — console.groq.com   14,400 req/day free; fastest inference
  TOGETHER_API_KEY    — api.together.xyz   free credits on signup; Llama/Mixtral
  GEMINI_API_KEY      — aistudio.google.com  free tier via Gemini API (Google AI)
  MISTRAL_API_KEY     — console.mistral.ai   free tier on La Plateforme

Design rules:
  - NEVER blocks the main trading loop — all calls run in ThreadPoolExecutor
  - NEVER changes trade approval — advisory text only
  - Degrades silently to {} if keys are missing or requests unavailable
  - TOTAL_TIMEOUT_S = 10s wall-clock across all parallel calls
  - Each model independently ignorant of others — genuine diverse perspectives

Keys go in .env (never commit):
  OPENROUTER_API_KEY=sk-or-...
  GROQ_API_KEY=gsk_...
  TOGETHER_API_KEY=...
  GEMINI_API_KEY=...
  MISTRAL_API_KEY=...
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

logger = logging.getLogger(__name__)

# ── Provider endpoints ─────────────────────────────────────────────────────────

_OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
_GROQ_BASE       = "https://api.groq.com/openai/v1/chat/completions"
_TOGETHER_BASE   = "https://api.together.xyz/v1/chat/completions"
_GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_MISTRAL_BASE    = "https://api.mistral.ai/v1/chat/completions"
_KIMI_BASE       = "https://api.moonshot.cn/v1/chat/completions"

# ── Model registry ─────────────────────────────────────────────────────────────
# Add/remove entries here to change which models get called.
# "key_env" = environment variable that must be set for this model to activate.
# "free" = True means no per-token cost on the provider's free tier.

ALL_MODELS: list[dict] = [
    # OpenRouter — routes to best available free model
    {
        "id":       "qwen/qwen3-8b:free",
        "endpoint": _OPENROUTER_BASE,
        "key_env":  "OPENROUTER_API_KEY",
        "label":    "Qwen3-8B",
        "free":     True,
        "extra_headers": {"HTTP-Referer": "https://aiden-trading-bot", "X-Title": "AiDEN"},
    },
    {
        "id":       "meta-llama/llama-3.3-70b-instruct:free",
        "endpoint": _OPENROUTER_BASE,
        "key_env":  "OPENROUTER_API_KEY",
        "label":    "Llama3.3-70B",
        "free":     True,
        "extra_headers": {"HTTP-Referer": "https://aiden-trading-bot", "X-Title": "AiDEN"},
    },
    {
        "id":       "google/gemma-3-27b-it:free",
        "endpoint": _OPENROUTER_BASE,
        "key_env":  "OPENROUTER_API_KEY",
        "label":    "Gemma3-27B",
        "free":     True,
        "extra_headers": {"HTTP-Referer": "https://aiden-trading-bot", "X-Title": "AiDEN"},
    },
    # Groq — fastest inference, free daily quota
    {
        "id":       "llama-3.3-70b-versatile",
        "endpoint": _GROQ_BASE,
        "key_env":  "GROQ_API_KEY",
        "label":    "Groq/Llama3.3-70B",
        "free":     True,
        "extra_headers": {},
    },
    {
        "id":       "mixtral-8x7b-32768",
        "endpoint": _GROQ_BASE,
        "key_env":  "GROQ_API_KEY",
        "label":    "Groq/Mixtral-8x7B",
        "free":     True,
        "extra_headers": {},
    },
    # Together AI — Llama variants
    {
        "id":       "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "endpoint": _TOGETHER_BASE,
        "key_env":  "TOGETHER_API_KEY",
        "label":    "Together/Llama3.3-70B",
        "free":     True,
        "extra_headers": {},
    },
    # Google Gemini via OpenAI-compatible endpoint
    {
        "id":       "gemini-2.0-flash",
        "endpoint": _GEMINI_BASE,
        "key_env":  "GEMINI_API_KEY",
        "label":    "Gemini-2.0-Flash",
        "free":     True,
        "extra_headers": {},
    },
    # Mistral — free tier on La Plateforme
    {
        "id":       "mistral-small-latest",
        "endpoint": _MISTRAL_BASE,
        "key_env":  "MISTRAL_API_KEY",
        "label":    "Mistral-Small",
        "free":     True,
        "extra_headers": {},
    },
    # Kimi (Moonshot AI) — long-context specialist, strong reasoning
    {
        "id":       "moonshot-v1-8k",
        "endpoint": _KIMI_BASE,
        "key_env":  "KIMI_API_KEY",
        "label":    "Kimi",
        "free":     False,
        "extra_headers": {},
    },
]

# For trade advisory: use only 2-3 fast models — don't spam all 8 per trade
_TRADE_ADVISORY_MODELS = ["Qwen3-8B", "Groq/Llama3.3-70B", "Gemini-2.0-Flash"]

# Global wall-clock ceiling for ALL parallel calls
TOTAL_TIMEOUT_S = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def _have_requests() -> bool:
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def _active_models(labels: list[str] | None = None) -> list[dict]:
    """Return models that have a key set, optionally filtered to label list."""
    pool = ALL_MODELS if labels is None else [m for m in ALL_MODELS if m["label"] in labels]
    return [m for m in pool if os.environ.get(m["key_env"], "").strip()]


def _call_model(
    model: dict, prompt: str, max_tokens: int = 200,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Synchronous single-model call. Returns (label, text_or_error)."""
    import requests

    key = os.environ.get(model["key_env"], "")
    if not key:
        return model["label"], "[key not set]"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        **model.get("extra_headers", {}),
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":       model["id"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.4,
    }
    try:
        resp = requests.post(model["endpoint"], json=payload, headers=headers, timeout=9)
        if resp.status_code != 200:
            return model["label"], f"[HTTP {resp.status_code}: {resp.text[:120]}]"
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return model["label"], text or "[empty]"
    except Exception as exc:
        return model["label"], f"[error: {exc}]"


# ── Public API ─────────────────────────────────────────────────────────────────

def query(
    prompt: str,
    model_labels: list[str] | None = None,
    max_tokens: int = 200,
    timeout: float = TOTAL_TIMEOUT_S,
    system_prompt: str | None = None,
) -> dict[str, str]:
    """Query all configured (or specified) models concurrently.

    Args:
        prompt:       The prompt to send to every model.
        model_labels: If given, restrict to these model labels. None = all active.
        max_tokens:   Token cap per model response.
        timeout:      Wall-clock deadline across all models (seconds).

    Returns:
        {model_label: response_text} — only for models with keys set.
        {} if requests not installed or no keys configured.
    """
    if not _have_requests():
        return {}

    active = _active_models(model_labels)
    if not active:
        return {}

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {pool.submit(_call_model, m, prompt, max_tokens, system_prompt): m["label"] for m in active}
        try:
            for fut in as_completed(futures, timeout=timeout):
                label, text = fut.result()
                results[label] = text
        except FuturesTimeout:
            for fut, label in futures.items():
                if label not in results:
                    results[label] = "[timeout]"

    return results


def trade_advisory(
    symbol: str,
    direction: int,
    score: int,
    daily_dd_pct: float,
) -> dict[str, str]:
    """Single-sentence risk perspective on a pending trade from 2-3 fast models."""
    dir_str = "LONG" if direction > 0 else "SHORT"
    prompt = (
        f"AiDEN FTMO $100k trading bot — pending trade:\n"
        f"Symbol: {symbol} | Direction: {dir_str} | Score: {score}/10 | "
        f"Daily drawdown so far: {daily_dd_pct:.2f}%\n\n"
        f"As a senior trading risk advisor, give ONE concise sentence on the most "
        f"important risk consideration for this trade. Be direct. No preamble."
    )
    return query(prompt, model_labels=_TRADE_ADVISORY_MODELS, max_tokens=120)


def ask(prompt: str, model_labels: list[str] | None = None) -> dict[str, str]:
    """General-purpose multi-model query. Used from CLI and external callers."""
    return query(prompt, model_labels=model_labels, max_tokens=800, timeout=30)


def list_models() -> list[dict]:
    """Return all models with their active status."""
    out = []
    for m in ALL_MODELS:
        key_set = bool(os.environ.get(m["key_env"], "").strip())
        out.append({
            "label":    m["label"],
            "id":       m["id"],
            "provider": m["key_env"].replace("_API_KEY", ""),
            "free":     m["free"],
            "active":   key_set,
        })
    return out


# ── CLI interface ──────────────────────────────────────────────────────────────
# Usage:
#   python -m execution.llm_advisor "What is your view on XAUUSD right now?"
#   python -m execution.llm_advisor --models "Groq/Llama3.3-70B,Gemini-2.0-Flash" "Explain FVG"
#   python -m execution.llm_advisor --list

def _load_env() -> None:
    """Load .env from project root into os.environ (setdefault — never overwrite)."""
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _default_labels() -> list[str] | None:
    """Return labels from DEFAULT_LLM_MODELS env var, or None (= all active)."""
    raw = os.environ.get("DEFAULT_LLM_MODELS", "").strip()
    if not raw:
        return None
    return [l.strip() for l in raw.split(",") if l.strip()]


def _print_models() -> None:
    models = list_models()
    print(f"\n{'Label':<26} {'Provider':<16} {'Active'}")
    print("-" * 55)
    for m in models:
        tick = "ACTIVE" if m["active"] else "  ---"
        print(f"{m['label']:<26} {m['provider']:<16} {tick}")
    active_count = sum(1 for m in models if m["active"])
    print(f"\n{active_count}/{len(models)} active — set keys in .env to activate more\n")


def _print_results(results: dict[str, str]) -> None:
    if not results:
        print("\nNo responses — set at least one API key in .env\n")
        return
    for label, text in results.items():
        print(f"\n{'='*58}")
        print(f"  {label}")
        print(f"{'='*58}")
        print(text)
    print()


def _repl_mode(system_prompt: str | None = None) -> None:
    """Interactive REPL — start once, keep querying without retyping commands."""
    _load_env()
    active = _active_models()
    if not active:
        print("\nNo models active. Add at least one API key to .env first.\n")
        return

    current_labels: list[str] | None = _default_labels()
    label_str = ", ".join(current_labels) if current_labels else "all active"
    mode_tag = " [CODER MODE]" if system_prompt else ""

    print(f"\nAiDEN Multi-Model REPL{mode_tag}  (models: {label_str})")
    print("Commands:  /models  /switch <label>  /all  /reset  /coder  /exit")
    print("Shortcuts: g=Groq  gem=Gemini  k=Kimi  m=Mistral  t=Together  o=OpenRouter")
    print("-" * 58)

    shortcuts = {
        "g":       ["Groq/Llama3.3-70B", "Groq/Mixtral-8x7B"],
        "groq":    ["Groq/Llama3.3-70B", "Groq/Mixtral-8x7B"],
        "o":       ["Qwen3-8B", "Llama3.3-70B", "Gemma3-27B"],
        "or":      ["Qwen3-8B", "Llama3.3-70B", "Gemma3-27B"],
        "gem":     ["Gemini-2.0-Flash"],
        "gemini":  ["Gemini-2.0-Flash"],
        "m":       ["Mistral-Small"],
        "mistral": ["Mistral-Small"],
        "t":       ["Together/Llama3.3-70B"],
        "together":["Together/Llama3.3-70B"],
        "k":       ["Kimi"],
        "kimi":    ["Kimi"],
    }

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw in ("/exit", "/quit", "exit", "quit"):
            print("Exiting.")
            break

        if raw == "/models":
            _print_models()
            continue

        if raw == "/coder":
            from execution.aiden_context import build_system_prompt
            system_prompt = build_system_prompt()
            print("[Coder mode ON — AiDEN context injected as system prompt]")
            continue

        if raw == "/nocoder":
            system_prompt = None
            print("[Coder mode OFF]")
            continue

        if raw == "/all":
            current_labels = None
            print("Switched to: all active models")
            continue

        if raw == "/reset":
            current_labels = _default_labels()
            label_str = ", ".join(current_labels) if current_labels else "all active"
            print(f"Reset to default: {label_str}")
            continue

        if raw.startswith("/switch "):
            target = raw[8:].strip()
            if target in shortcuts:
                current_labels = shortcuts[target]
            else:
                # Try as a direct label match
                names = [m["label"] for m in ALL_MODELS]
                matches = [n for n in names if target.lower() in n.lower()]
                if matches:
                    current_labels = matches
                else:
                    print(f"Unknown model: {target}. Use /models to see labels.")
                    continue
            label_str = ", ".join(current_labels)
            print(f"Switched to: {label_str}")
            continue

        # Check for inline prefix: "g: question" or "gem: question"
        prompt = raw
        labels_for_this = current_labels
        for sc, sc_labels in shortcuts.items():
            if raw.lower().startswith(f"{sc}: ") or raw.lower().startswith(f"{sc}:"):
                prefix_len = len(sc) + 1 + (1 if raw[len(sc)+1:len(sc)+2] == " " else 0)
                prompt = raw[prefix_len:].strip()
                labels_for_this = sc_labels
                break

        if not prompt:
            continue

        label_str_now = ", ".join(labels_for_this) if labels_for_this else "all active"
        print(f"[{label_str_now}] ...", end="", flush=True)
        results = query(prompt, model_labels=labels_for_this, max_tokens=800, timeout=30,
                        system_prompt=system_prompt)
        print("\r", end="")
        _print_results(results)


def _cli_main() -> None:
    import argparse
    _load_env()

    parser = argparse.ArgumentParser(
        prog="python -m execution.llm_advisor",
        description="Query multiple free AI models — or start an interactive REPL session.",
        epilog=(
            "Model shortcuts (use as prefix in prompt): "
            "g:=Groq  o:=OpenRouter  gem:=Gemini  m:=Mistral  t:=Together  all:=all"
        ),
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send (omit for REPL mode)")
    parser.add_argument("--models", "-m", default=None,
                        help="Comma-separated model labels (default: DEFAULT_LLM_MODELS or all active)")
    parser.add_argument("--list", "-l", action="store_true", help="List models and exit")
    parser.add_argument("--repl", "-r", action="store_true", help="Interactive REPL session")
    parser.add_argument("--coder", "-c", action="store_true",
                        help="Inject full AiDEN codebase context as system prompt (coding assistant mode)")
    parser.add_argument("--handoff", action="store_true",
                        help="Export AiDEN context handoff document (paste into any AI when Claude runs out)")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    if args.list:
        _print_models()
        return

    if args.handoff:
        from execution.aiden_context import export_handoff
        from pathlib import Path
        out = Path("logs") / "handoff.md"
        doc = export_handoff(out)
        print(doc)
        print(f"\n[Saved to {out}]")
        return

    sys_prompt = None
    if args.coder:
        from execution.aiden_context import build_system_prompt
        sys_prompt = build_system_prompt()
        print("[Coder mode: AiDEN context loaded as system prompt]\n")

    if args.repl or not args.prompt:
        _repl_mode(system_prompt=sys_prompt)
        return

    # Detect inline prefix shortcut
    shortcuts = {
        "g:": ["Groq/Llama3.3-70B", "Groq/Mixtral-8x7B"],
        "groq:": ["Groq/Llama3.3-70B", "Groq/Mixtral-8x7B"],
        "o:": ["Qwen3-8B", "Llama3.3-70B", "Gemma3-27B"],
        "or:": ["Qwen3-8B", "Llama3.3-70B", "Gemma3-27B"],
        "gem:": ["Gemini-2.0-Flash"],
        "gemini:": ["Gemini-2.0-Flash"],
        "m:": ["Mistral-Small"],
        "mistral:": ["Mistral-Small"],
        "t:": ["Together/Llama3.3-70B"],
        "together:": ["Together/Llama3.3-70B"],
        "k:": ["Kimi"],
        "kimi:": ["Kimi"],
        "all:": None,
    }
    prompt = args.prompt
    labels = [l.strip() for l in args.models.split(",")] if args.models else _default_labels()
    for prefix, sc_labels in shortcuts.items():
        if prompt.lower().startswith(prefix):
            prompt = prompt[len(prefix):].strip()
            labels = sc_labels
            break

    active = _active_models(labels)
    print(f"\nQuerying {len(active)} model(s)...\n")
    results = query(prompt, model_labels=labels, max_tokens=args.max_tokens,
                    timeout=args.timeout, system_prompt=sys_prompt)
    _print_results(results)


if __name__ == "__main__":
    _cli_main()
