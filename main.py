"""
MLTrader - hybrid pipeline: technical ML model + Gemini research agent.

Usage:
    python main.py          # trades SPY (default)
    python main.py AAPL     # any symbol

Signal combining rule:
    ML model is the primary driver.
    LLM acts as a filter: it can VETO a trade if it strongly disagrees
    (opposite signal with confidence >= LLM_VETO_CONFIDENCE).
    No longer requires LLM agreement to act -- only LLM opposition blocks a trade.
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
DEFAULT_SYMBOL      = "SPY"
SHARES_PER_TRADE    = 1       # paper-account sizing (coursework used 1000-share lots)
IMPACT              = 0.005   # estimated round-trip market impact (0.5%)
TRAIN_YEARS         = 2       # years of history used to train the ML model
TEST_YEARS          = 1       # years fed to testPolicy to arrive at today's signal
LOOKAHEAD_DAYS      = 20      # label lookahead window (20 days reduces noise vs 5)
LLM_VETO_CONFIDENCE = 0.7    # LLM blocks a trade only if it strongly disagrees at this level

# ETFs / broad-market indexes: long-bias mode + no shorting
LONG_BIAS_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "GLD"}

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
    """Train and run the ML model, returning the most recent trade signal."""
    now         = datetime.now(ZoneInfo("America/New_York"))
    train_start = now - timedelta(days=365 * TRAIN_YEARS)
    train_end   = now - timedelta(days=365 * TEST_YEARS)
    test_start  = train_end
    test_end    = now - timedelta(days=1)

    long_bias = symbol.upper() in LONG_BIAS_SYMBOLS

    print(f"\n[{symbol}] Training  : {train_start.date()} -> {train_end.date()}")
    learner.add_evidence(
        symbol=symbol, sd=train_start, ed=train_end, lookahead=LOOKAHEAD_DAYS
    )

    print(f"[{symbol}] Signaling : {test_start.date()} -> {test_end.date()} "
          f"(long_bias={long_bias})")
    trades = learner.testPolicy(
        symbol=symbol, sd=test_start, ed=test_end,
        long_bias=long_bias, lookahead=LOOKAHEAD_DAYS
    )
    print(f"\nLast 10 trade signals:\n{trades.tail(10)}\n")

    return float(trades["shares"].iloc[-1])


def combine_signals(symbol: str, tech_signal: float, llm_rec: dict) -> int:
    """
    ML model drives; LLM acts as a selective veto.

    - If ML says buy/sell: execute UNLESS LLM strongly disagrees
      (opposite direction AND confidence >= LLM_VETO_CONFIDENCE).
    - If ML says hold (0): no trade regardless of LLM.
    - Never short LONG_BIAS_SYMBOLS (ETFs).

    Returns desired share delta: +1000, -1000, or 0 (hold).
    """
    tech_dir = 1 if tech_signal > 0 else (-1 if tech_signal < 0 else 0)

    # No signal from the technical model -> stay put
    if tech_dir == 0:
        return 0

    # Never short ETFs / broad-market indexes
    if symbol.upper() in LONG_BIAS_SYMBOLS and tech_dir < 0:
        return 0

    llm_dir = (
         1 if llm_rec["signal"] == "buy"  else
        -1 if llm_rec["signal"] == "sell" else
         0
    )
    confidence = llm_rec["confidence"]

    # LLM veto: strongly opposing signal
    if llm_dir != 0 and llm_dir != tech_dir and confidence >= LLM_VETO_CONFIDENCE:
        print(f"  [veto] LLM overrides ML signal "
              f"(LLM={llm_rec['signal']}, conf={confidence:.2f})")
        return 0

    return tech_dir * 1000


def execute_trade(symbol: str, desired_delta: int, current_qty: int) -> None:
    """
    Convert a desired position delta into Alpaca paper orders.

        +1000 -> go long   (buy if flat)
        -1000 -> go short  (sell if flat, or close long)
            0 -> hold
    """
    desired_dir = 1 if desired_delta > 0 else (-1 if desired_delta < 0 else 0)
    current_dir = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

    if desired_dir == current_dir:
        state = "long" if desired_dir > 0 else "short" if desired_dir < 0 else "flat"
        print(f"[{symbol}] No trade - already {state}.")
        return

    # Close any existing position first
    if current_dir == 1:
        order  = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.SELL, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Closed long  -> order {result.id}")
    elif current_dir == -1:
        order  = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.BUY, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Closed short -> order {result.id}")

    # Open new position
    if desired_dir == 1:
        order  = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.BUY, TimeInForce.DAY)
        result = submitOrder(order)
        print(f"[{symbol}] Opened long  -> order {result.id}")
    elif desired_dir == -1:
        order  = prepareOrder(symbol, SHARES_PER_TRADE, OrderSide.SELL, TimeInForce.DAY)
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
    desired_delta = combine_signals(symbol, tech_signal, llm_rec)
    action = (
        "BUY"  if desired_delta > 0 else
        "SELL" if desired_delta < 0 else
        "HOLD"
    )
    print(f"\n[{symbol}] Final decision : {action}")

    # 5. Execute
    execute_trade(symbol, desired_delta, current_qty)


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_SYMBOL
    main(sym)
