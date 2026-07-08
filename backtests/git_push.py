"""Auto-commit and push all changes to GitHub after any major run."""
from __future__ import annotations
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def auto_push(message: str = "Auto: backtest run", push: bool = True) -> bool:
    """Stage all changes, commit, and push to origin/master.

    Returns True if push succeeded, False on any error.
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    commit_msg = f"{message} [{ts}]"

    def run(cmd: list[str]) -> tuple[int, str]:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return result.returncode, (result.stdout + result.stderr).strip()

    # Stage everything (logs, results, new/modified files)
    code, out = run(["git", "add", "-A"])
    if code != 0:
        print(f"  [git] add failed: {out}")
        return False

    # Check if there's anything to commit
    code, out = run(["git", "diff", "--cached", "--quiet"])
    if code == 0:
        print("  [git] Nothing to commit.")
        return True

    # Commit
    code, out = run(["git", "commit", "-m", commit_msg])
    if code != 0:
        print(f"  [git] commit failed: {out}")
        return False
    print(f"  [git] Committed: {commit_msg}")

    if not push:
        return True

    # Push
    code, out = run(["git", "push", "origin", "master"])
    if code != 0:
        print(f"  [git] push failed: {out}")
        return False

    print(f"  [git] Pushed to GitHub.")
    return True
