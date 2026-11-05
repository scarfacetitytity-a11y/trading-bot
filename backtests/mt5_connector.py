"""Read-only MetaTrader 5 connection helpers.

This module only attaches to an already-running, already-logged-in MT5
terminal and reports its status. It never places trades and never attempts
to log in with credentials.
"""
import logging
import os
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

# Set by connect() on every call; callers can inspect to distinguish error classes.
_last_connect_error: Optional[int] = None

_TRADE_MODE_NAMES = {
    mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
    mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
    mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
}


def connect(terminal_path: Optional[str] = None) -> bool:
    """Attach to a running MT5 terminal and verify it is logged in.

    Returns True if connected to a logged-in account, False otherwise.
    Always logs the reason on failure.
    """
    global _last_connect_error
    _last_connect_error = None

    initialized = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()

    if not initialized:
        err = mt5.last_error()
        err_code = err[0]
        _last_connect_error = err_code

        # -6 = Authorization failed: terminal is open but session expired.
        # Retry with credentials from env if available.
        if err_code == -6:
            _login = os.environ.get("MT5_LOGIN")
            _password = os.environ.get("MT5_PASSWORD", "")
            _server = os.environ.get("MT5_SERVER", "")
            if _login:
                logger.warning(
                    "MT5 initialize() failed with authorization error — retrying with credentials for account %s",
                    _login,
                )
                kwargs = dict(login=int(_login), password=_password, server=_server)
                if terminal_path:
                    kwargs["path"] = terminal_path
                initialized = mt5.initialize(**kwargs)
                if initialized:
                    logger.info("MT5 re-initialized with credentials for account %s", _login)
                else:
                    logger.error(
                        "MT5 credential retry also failed: %s — "
                        "session may require a manual login in the MT5 terminal.",
                        mt5.last_error(),
                    )
            else:
                logger.error(
                    "MT5 authorization failed (-6) and no MT5_LOGIN env var is set — "
                    "set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER or manually log into the terminal."
                )

        # -10001 = IPC send failed: a stale IPC handle is open from a previous
        # crashed terminal session. Shut down to release it so the next connect()
        # call starts with a clean pipe.
        elif err_code == -10001:
            logger.error(
                "MT5 IPC send failed (-10001) — stale IPC handle detected. "
                "Releasing it so the next reconnect attempt starts clean."
            )
            mt5.shutdown()

        # -10003 = IPC process create failed: MT5 could not launch the terminal
        # executable. Common causes: wrong path, zombie process holding the lock,
        # or UAC/permissions blocking the launch.
        # Calling shutdown() releases any partial IPC state so the next retry
        # attempt starts clean (mirrors the -10001 handling above).
        elif err_code == -10003:
            _path_used = terminal_path or "<default MT5 path>"
            logger.error(
                "MT5 cannot launch terminal process (-10003) at path: %s — "
                "open Task Manager and kill any existing terminal64.exe, then "
                "open the MT5 terminal manually before restarting the bot.",
                _path_used,
            )
            mt5.shutdown()

    if not initialized:
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        logger.error(
            "Checklist: MT5 terminal installed? Terminal open? "
            "'terminal_path' in config.yaml correct? Logged into an account?"
        )
        return False

    # If MT5_LOGIN is set in the environment, ensure the terminal is on that
    # account — switch via mt5.login() when it isn't.
    _env_login = os.environ.get("MT5_LOGIN")
    if _env_login:
        _acct = mt5.account_info()
        if _acct is None or str(_acct.login) != str(_env_login):
            _ok = mt5.login(
                int(_env_login),
                password=os.environ.get("MT5_PASSWORD", ""),
                server=os.environ.get("MT5_SERVER", ""),
            )
            if not _ok:
                logger.error("MT5 login(%s) failed: %s", _env_login, mt5.last_error())
                mt5.shutdown()
                return False
            logger.info("Switched MT5 terminal to account %s", _env_login)

    terminal = mt5.terminal_info()
    if terminal is None:
        logger.error("Could not read MT5 terminal info: %s", mt5.last_error())
        mt5.shutdown()
        return False

    if not terminal.connected:
        logger.error("MT5 terminal is open but not connected to a trade server.")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    if account is None:
        logger.error(
            "MT5 terminal is connected but no account is logged in. "
            "Open MT5 and log into a demo account."
        )
        mt5.shutdown()
        return False

    mode = _TRADE_MODE_NAMES.get(account.trade_mode, f"UNKNOWN({account.trade_mode})")
    logger.info(
        "Connected to MT5 | account=%s | server=%s | mode=%s | terminal=%s build %s",
        account.login, account.server, mode, terminal.name, terminal.build,
    )

    if mode != "DEMO":
        logger.warning(
            "Account trade mode is '%s', not DEMO. This pipeline is read-only "
            "and will not place trades, but confirm this is the account you intend to use.",
            mode,
        )

    return True


def disconnect() -> None:
    """Close the MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed.")
