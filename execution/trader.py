"""MT5 order execution and position management.

Handles placing, closing, and inspecting positions for the bot.
All bot orders are tagged with a magic number so they can be
distinguished from manual trades in the terminal.
"""
import logging

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

# Filled once at startup from config
_MAGIC: int = 234001


def set_magic(magic: int) -> None:
    global _MAGIC
    _MAGIC = magic


# ---------------------------------------------------------------------------
# Account / market info
# ---------------------------------------------------------------------------

def get_account() -> dict:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Could not read account info: {mt5.last_error()}")
    return {"balance": info.balance, "equity": info.equity, "margin_free": info.margin_free}


def get_tick(symbol: str) -> mt5.Tick:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Could not get tick for {symbol}: {mt5.last_error()}")
    return tick


# ---------------------------------------------------------------------------
# Position queries
# ---------------------------------------------------------------------------

def get_positions(symbol: str) -> list:
    """Return all open bot positions for this symbol (filtered by magic)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == _MAGIC]


def get_all_positions() -> list:
    """Return all open bot positions across every symbol (filtered by magic)."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [p for p in positions if p.magic == _MAGIC]


def get_position_direction(symbol: str) -> int:
    """Return 1 (long), -1 (short), or 0 (flat) for the bot's position."""
    positions = get_positions(symbol)
    if not positions:
        return 0
    pos = positions[0]
    return 1 if pos.type == mt5.ORDER_TYPE_BUY else -1


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_order(
    symbol: str,
    direction: int,
    lots: float,
    sl: float | None = None,
    tp: float | None = None,
    comment: str = "trading-bot",
) -> bool:
    """Send a market order. direction: 1=buy, -1=sell. Returns True on success."""
    tick = get_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == 1 else tick.bid

    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error("Symbol info unavailable for %s", symbol)
        return False

    type_filling = _best_filling_mode(info)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": round(lots, 2),
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": _MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }
    if sl:
        request["sl"] = round(sl, info.digits)
    if tp:
        request["tp"] = round(tp, info.digits)

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        comment_out = result.comment if result else mt5.last_error()
        logger.error(
            "Order FAILED: %s %s %.2f lots | retcode=%s | %s",
            "BUY" if direction == 1 else "SELL", symbol, lots, code, comment_out,
        )
        return False

    logger.info(
        "Order OK: %s %s %.2f lots @ %.5f | SL=%s TP=%s | ticket=%s",
        "BUY" if direction == 1 else "SELL", symbol, lots, price, sl, tp, result.order,
    )
    return True


def close_position(position) -> bool:
    """Close a single MT5 position object. Returns True on success."""
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = get_tick(position.symbol)
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    info = mt5.symbol_info(position.symbol)
    type_filling = _best_filling_mode(info) if info else mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": _MAGIC,
        "comment": "close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.error("Close FAILED: ticket=%s | retcode=%s", position.ticket, code)
        return False

    logger.info("Closed position: ticket=%s %s", position.ticket, position.symbol)
    return True


def close_all(symbol: str) -> int:
    """Close all bot positions for a symbol. Returns number closed."""
    closed = 0
    for pos in get_positions(symbol):
        if close_position(pos):
            closed += 1
    return closed


def partial_close(position, pct: float) -> bool:
    """Close pct fraction (0–1) of a position by volume. Returns True on success."""
    vol = round(position.volume * pct, 2)
    if vol < 0.01:
        logger.warning("partial_close: rounded volume %.2f too small, skipping", vol)
        return False

    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = get_tick(position.symbol)
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    info = mt5.symbol_info(position.symbol)
    type_filling = _best_filling_mode(info) if info else mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": vol,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": _MAGIC,
        "comment": "partial_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.error("partial_close FAILED: ticket=%s pct=%.0f%% retcode=%s",
                     position.ticket, pct * 100, code)
        return False

    logger.info("partial_close OK: ticket=%s pct=%.0f%% vol=%.2f",
                position.ticket, pct * 100, vol)
    return True


def modify_sl_tp(
    symbol: str,
    ticket: int,
    new_sl: float | None = None,
    new_tp: float | None = None,
) -> bool:
    """Modify SL and/or TP of an open position via TRADE_ACTION_SLTP."""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error("modify_sl_tp: no symbol info for %s", symbol)
        return False

    # Guard: broker requires SL/TP >= stop_level points from current price.
    # Skipping instead of sending prevents retcode 10016 (invalid stops).
    if new_sl is not None:
        tick     = mt5.symbol_info_tick(symbol)
        stop_pts = getattr(info, "trade_stops_level", 0) or 0
        min_dist = stop_pts * (getattr(info, "point", 0.00001) or 0.00001)
        if tick is not None and min_dist > 0:
            sl_above = new_sl > tick.bid
            dist = (new_sl - tick.ask) if sl_above else (tick.bid - new_sl)
            if dist < min_dist:
                logger.debug("modify_sl_tp: SL %.5f within stop_level %.5f of price — skip",
                             new_sl, min_dist)
                return False

    request: dict = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": ticket}
    if new_sl is not None:
        request["sl"] = round(new_sl, info.digits)
    if new_tp is not None:
        request["tp"] = round(new_tp, info.digits)

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.error("modify_sl_tp FAILED: ticket=%s retcode=%s", ticket, code)
        return False

    logger.info("modify_sl_tp OK: ticket=%s sl=%s tp=%s", ticket, new_sl, new_tp)
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _best_filling_mode(info) -> int:
    """Return the best supported filling mode for this symbol/broker."""
    mode = getattr(info, "filling_mode", 0)
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN
