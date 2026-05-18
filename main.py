"""
MLTrader -hybrid pipeline: technical ML model + Claude research agent.

Usage:
    python main.py          # trades SPY (default)
    python main.py AAPL     # any symbol

Signal combining rule:
    Both signals must agree AND LLM confidence >= MIN_LLM_CONFIDENCE to act.
    If they disagree, or LLM is uncertain -> hold.
"""

import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from StrategyLearner import StrategyLearner
from researcher import research
from placeOrder import prepareOrder, submitOrder

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SYMBOL     = "SPY"
SHARES_PER_TRADE   = 1      # paper-account sizing (coursework used 1000-share lots)
IMPACT             = 0.005  # estimated round-trip market impact (0.5 %)
TRAIN_YEARS        = 2      # years of history used to train the ML model
TEST_YEARS         = 1      # years fed to testPolicy to arrive at today's signal
MIN_LLM_CONFIDENCE = 0.6   # LLM must be at least this confident to act

_trading_client = TradingClient(config.public, config.secret, paper=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_current_position(symbol: str) -> int:
    """Return share count currently held (0 if flat, negative if short)."""
    try:
        pos = _trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def get_technical_signal(symbol: str, learner: StrategyLearner) -> float:
    """
    Run the ML model over the test window and return the most recent signal.
    Also returns the latest indicator values for use as LLM context.
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    train_start = now - timedelta(days=365 * TRAIN_YEARS)
    train_end   = now - timedelta(days=365 * TEST_YEARS)
    test_start  = train_end
    test_end    = now - timedelta(days=1)

    print(f"\n[{symbol}] Training  : {train_start.date()} -> {train_end.date()}")
    learner.add_evidence(symbol=symbol, sd=train_start, ed=train_end)

    print(f"[{symbol}] Signaling : {test_start.date()} -> {test_end.date()}")
    trades = learner.testPolicy(symbol=symbol, sd=test_start, ed=test_end)
    print(f"\nLast 10 trade signals:\n{trades.tail(10)}\n")

    return float(trades["shares"].iloc[-1])


def combine_signals(tech_signal: float, llm_rec: dict) -> int:
    """
    Conservative combination: both must agree AND LLM confidence must meet threshold.
    Returns desired share delta: +1000, -1000, or 0 (hold).
    """
    tech_dir = 1 if tech_signal > 0 else (-1 if tech_signal < 0 else 0)
    llm_dir  = (
        1  if llm_rec["signal"] == "buy"  else
        -1 if llm_rec["signal"] == "sell" else
        0
    )

    if tech_dir == llm_dir and llm_rec["confidence"] >= MIN_LLM_CONFIDENCE and tech_dir != 0:
        return tech_dir * 1000
    return 0


def execute_trade(symbol: str, desired_delta: int, current_qty: int) -> None:
    """
    Convert a desired position delta into Alpaca paper orders.

    desired_delta:
        +1000 -> go long   (buy if flat, buy-to-cover + buy if short)
        -1000 -> go short  (sell if flat, sell + short if long)
            0 -> hold / close to flat
    """
    desired_dir = 1 if desired_delta > 0 else (-1 if desired_delta < 0 else 0)
    current_dir = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

    if desired_dir == current_dir:
        state = "long" if desired_dir > 0 else "short" if desired_dir < 0 else "flat"
        print(f"[{symbol}] No trade - already {state}.")
        return

    # Close any existing position first
    if current_dir == 1:
        order = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.SELL, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Closed long  -> order {result.id}")
    elif current_dir == -1:
        order = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.BUY, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Closed short -> order {result.id}")

    # Open new position (skip if targeting flat)
    if desired_dir == 1:
        order = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.BUY, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Opened long  -> order {result.id}")
    elif desired_dir == -1:
        order = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.SELL, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Opened short -> order {result.id}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(symbol: str = DEFAULT_SYMBOL) -> None:
    print(f"=== MLTrader | {symbol} | {datetime.now().date()} ===")

    learner = StrategyLearner(verbose=True, impact=IMPACT)

    # 1. Technical signal + indicator snapshot
    tech_signal = get_technical_signal(symbol, learner)
    indicators  = learner.get_latest_indicators(symbol)

    # 2. Current portfolio exposure
    current_qty = get_current_position(symbol)

    print(f"[{symbol}] Technical signal : {tech_signal:+.0f}")
    print(f"[{symbol}] Current position : {current_qty:+d} shares")
    print(f"[{symbol}] Latest indicators: {indicators}")

    # 3. LLM research
    print(f"\n[{symbol}] Running LLM research...")
    llm_rec = research(
        symbol=symbol,
        technical_signal=tech_signal,
        indicators=indicators,
        portfolio_qty=current_qty,
    )
    print(f"[{symbol}] LLM signal     : {llm_rec['signal'].upper()}")
    print(f"[{symbol}] LLM confidence : {llm_rec['confidence']:.2f}")
    print(f"[{symbol}] LLM reasoning  : {llm_rec['reasoning']}")

    # 4. Combine signals
    desired_delta = combine_signals(tech_signal, llm_rec)
    action = (
        "BUY"  if desired_delta > 0 else
        "SELL" if desired_delta < 0 else
        "HOLD (signals disagree or LLM not confident)"
    )
    print(f"\n[{symbol}] Final decision : {action}")

    # 5. Execute
    execute_trade(symbol, desired_delta, current_qty)


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_SYMBOL
    main(sym)
