"""MT5 order flow analysis — lightweight bookmap alternative.

Reads tick volume and DOM data from MT5 to surface:
- Volume delta (buying vs selling pressure)
- Absorption levels (high volume, small price move)
- Imbalance zones (one-sided volume spikes)
- DOM depth snapshot (bid/ask stacking)

Not a replacement for Bookmap — that needs a data feed subscription.
This uses what MT5 provides natively: tick volume, OHLC, DOM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowSnapshot:
    symbol:          str
    delta:           float      # positive = net buying, negative = net selling
    delta_pct:       float      # delta as % of total volume
    absorption:      bool       # high volume, tiny candle — institutional absorption
    imbalance_dir:   int        # +1 buy imbalance, -1 sell imbalance, 0 neutral
    dom_bid_depth:   float      # total volume stacked on bid side
    dom_ask_depth:   float      # total volume stacked on ask side
    dom_bias:        int        # +1 bid heavy (support), -1 ask heavy (resistance), 0 balanced
    volume_spike:    bool       # current bar volume > 2x recent average
    summary:         str        # human-readable one-liner


def analyse_order_flow(symbol: str, df_m5: pd.DataFrame, mt5=None) -> Optional[OrderFlowSnapshot]:
    """
    Analyse order flow for a symbol using M5 OHLCV bars and optional DOM.

    MT5 tick volume is not true delta — it counts ticks, not contracts.
    We proxy delta using the candle body: strong close near high = net buying.
    """
    try:
        if df_m5 is None or len(df_m5) < 20:
            return None

        recent = df_m5.iloc[-20:]
        bar    = df_m5.iloc[-1]

        high, low, open_, close = bar["high"], bar["low"], bar["open"], bar["close"]
        volume = bar.get("tick_volume", bar.get("volume", 0))

        candle_range = high - low
        if candle_range == 0:
            return None

        # Proxy delta: where did price close in the candle?
        # Close near top = buyers in control, close near bottom = sellers
        close_pct  = (close - low) / candle_range          # 0.0 = close at low, 1.0 = close at high
        body_pct   = abs(close - open_) / candle_range
        delta      = (close_pct - 0.5) * volume * 2        # positive = net buying pressure
        delta_pct  = (close_pct - 0.5) * 2                 # -1 to +1

        # Volume spike: current bar > 2x 20-bar average
        avg_vol    = recent["tick_volume"].mean() if "tick_volume" in recent.columns else 0
        vol_spike  = (avg_vol > 0) and (volume > 2 * avg_vol)

        # Absorption: high volume but tiny body — institutions absorbing supply/demand
        absorption = vol_spike and (body_pct < 0.25)

        # Imbalance: strong one-sided close with volume spike
        if delta_pct > 0.6 and vol_spike:
            imbalance_dir = 1
        elif delta_pct < -0.6 and vol_spike:
            imbalance_dir = -1
        else:
            imbalance_dir = 0

        # DOM depth — only available if mt5 module passed in
        dom_bid_depth = dom_ask_depth = 0.0
        dom_bias = 0
        if mt5 is not None:
            try:
                dom = mt5.market_book_get(symbol)
                if dom:
                    bids = [e for e in dom if e.type == mt5.BOOK_TYPE_BUY]
                    asks = [e for e in dom if e.type == mt5.BOOK_TYPE_SELL]
                    dom_bid_depth = sum(e.volume for e in bids)
                    dom_ask_depth = sum(e.volume for e in asks)
                    ratio = dom_bid_depth / (dom_bid_depth + dom_ask_depth + 1e-9)
                    if ratio > 0.6:
                        dom_bias = 1    # bid-heavy = buyers stacking = support
                    elif ratio < 0.4:
                        dom_bias = -1   # ask-heavy = sellers stacking = resistance
            except Exception:
                pass

        # Human summary
        direction  = "BUY" if delta_pct > 0.1 else "SELL" if delta_pct < -0.1 else "NEUTRAL"
        parts = [f"Flow: {direction} ({delta_pct:+.0%} delta)"]
        if absorption:
            parts.append("ABSORPTION (high vol, small body)")
        if imbalance_dir == 1:
            parts.append("BUY IMBALANCE")
        elif imbalance_dir == -1:
            parts.append("SELL IMBALANCE")
        if vol_spike:
            parts.append(f"VOL SPIKE ({volume:.0f} vs avg {avg_vol:.0f})")
        if dom_bias == 1:
            parts.append("DOM: bid-heavy (support)")
        elif dom_bias == -1:
            parts.append("DOM: ask-heavy (resistance)")

        return OrderFlowSnapshot(
            symbol        = symbol,
            delta         = delta,
            delta_pct     = delta_pct,
            absorption    = absorption,
            imbalance_dir = imbalance_dir,
            dom_bid_depth = dom_bid_depth,
            dom_ask_depth = dom_ask_depth,
            dom_bias      = dom_bias,
            volume_spike  = vol_spike,
            summary       = " | ".join(parts),
        )

    except Exception:
        logger.exception("[OrderFlow] analysis failed for %s", symbol)
        return None


@dataclass
class DOMLevel:
    price:     float
    volume:    float
    side:      str    # "bid" | "ask"
    strength:  float  # volume / avg_volume — how much bigger than normal this level is


def get_dom_key_levels(
    symbol: str,
    mt5,
    min_strength: float = 3.0,   # level must be 3x average DOM volume to qualify
    max_levels:   int   = 5,
) -> list[DOMLevel]:
    """Read live DOM and return price levels with unusually large order stacks.

    These are the 'bookmap walls' — where institutions have parked large orders.
    If 1000 lots sit at a price, that level is a magnet or a wall depending on
    which side of market it's on.

    Returns empty list if DOM unavailable (broker/symbol restriction).
    """
    if mt5 is None:
        return []
    try:
        if not mt5.market_book_add(symbol):
            return []
        dom = mt5.market_book_get(symbol)
        mt5.market_book_release(symbol)
        if not dom or len(dom) < 2:
            return []

        bids = [(e.price, e.volume) for e in dom if e.type == mt5.BOOK_TYPE_BUY and e.volume > 0]
        asks = [(e.price, e.volume) for e in dom if e.type == mt5.BOOK_TYPE_SELL and e.volume > 0]

        if not bids and not asks:
            return []

        all_vols = [v for _, v in bids + asks]
        avg_vol  = sum(all_vols) / len(all_vols) if all_vols else 1.0

        levels: list[DOMLevel] = []
        for price, vol in bids:
            strength = vol / avg_vol
            if strength >= min_strength:
                levels.append(DOMLevel(price=price, volume=vol, side="bid", strength=strength))
        for price, vol in asks:
            strength = vol / avg_vol
            if strength >= min_strength:
                levels.append(DOMLevel(price=price, volume=vol, side="ask", strength=strength))

        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:max_levels]

    except Exception:
        logger.debug("[DOM] key level scan failed for %s", symbol)
        return []


def order_flow_score_modifier(snap: Optional[OrderFlowSnapshot], signal_dir: int) -> int:
    """
    Returns a score modifier (-1, 0, +1) based on order flow alignment with signal.
    Used as an optional confluence input to the scoring engine.
    """
    if snap is None:
        return 0

    aligned = (snap.imbalance_dir == signal_dir) or (snap.dom_bias == signal_dir)
    opposing = (snap.imbalance_dir == -signal_dir) or (snap.dom_bias == -signal_dir)

    if snap.absorption:
        # Absorption is ambiguous — institutions could be accumulating to go either way
        return 0
    if aligned and snap.volume_spike:
        return 1
    if opposing and snap.volume_spike:
        return -1
    return 0
