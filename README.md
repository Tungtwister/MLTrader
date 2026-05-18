# MLTrader

A hybrid ML + LLM paper trading bot built as a continuation of Georgia Tech's CS 7646: Machine Learning for Trading.

The system combines a Bagged Random Tree technical model with a Gemini-powered research agent to generate trade signals, then executes them via the Alpaca paper trading API.

---

## How It Works

```
python main.py SPY
       |
       |-- StrategyLearner (Bagged Random Trees)
       |     Trains on 2 years of daily OHLCV data
       |     Features: SMA ratio, Bollinger %B, Momentum, PPO histogram
       |     Signals: +1000 (buy), -1000 (sell), 0 (hold)
       |
       |-- ResearchAgent (Gemini 2.5-flash-lite via REST API)
       |     Fetches live news from Alpaca News API
       |     Fetches revenue / net income / EPS from SEC EDGAR
       |     Returns: { signal, confidence, reasoning }
       |
       |-- Signal Combiner
       |     Both must agree AND LLM confidence >= 0.60 to act
       |     Otherwise: hold
       |
       +-- Alpaca Paper Trading API
             Places market orders on signal
```

---

## Project Structure

```
MLTrader/
|-- main.py              # Entry point — runs the full pipeline
|-- StrategyLearner.py   # Trains Bagged RT model, generates trade signals
|-- researcher.py        # Gemini LLM research agent (news + SEC data)
|-- indicators.py        # Technical indicators: SMA, BBP, Momentum, Stochastic, PPO
|-- RTLearner.py         # Random Tree decision tree (classification)
|-- BagLearner.py        # Bootstrap aggregating (100 bags)
|-- placeOrder.py        # Alpaca paper trading wrapper
|-- config.py            # API keys (gitignored — see Setup)
|-- requirements.txt     # Python dependencies
|-- Old/                 # Original GT coursework files (archive)
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create `config.py`** in the project root (this file is gitignored):
```python
public      = "YOUR_ALPACA_API_KEY"
secret      = "YOUR_ALPACA_SECRET_KEY"
gemini_key  = "YOUR_GEMINI_API_KEY"
```

- Alpaca keys: [alpaca.markets](https://alpaca.markets) — use a paper trading account
- Gemini key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free tier, no credit card required

---

## Usage

```bash
# Trade SPY (default)
python main.py

# Trade any symbol
python main.py AAPL
python main.py MSFT
```

**Example output:**
```
=== MLTrader | SPY | 2026-05-17 ===

[SPY] Training  : 2024-05-17 -> 2025-05-17
[StrategyLearner] Trained on 249 samples (SPY)
[SPY] Signaling : 2025-05-17 -> 2026-05-16

[SPY] Technical signal : +0
[SPY] Current position : +0 shares
[SPY] Latest indicators: SMA=0.028, BBP=0.782, MOM=0.088, PPO=0.065

[SPY] Running LLM research...
[SPY] LLM signal     : HOLD
[SPY] LLM confidence : 0.70
[SPY] LLM reasoning  : Mixed signals — strong earnings growth but price near upper Bollinger Band.

[SPY] Final decision : HOLD
[SPY] No trade - already flat.
```

---

## Signal Logic

| Technical | LLM | Confidence | Action |
|-----------|-----|------------|--------|
| BUY | buy | >= 0.60 | Execute buy |
| SELL | sell | >= 0.60 | Execute sell |
| Any | Any | < 0.60 | Hold |
| Disagree | — | — | Hold |

---

## Technical Indicators

| Indicator | Window | Interpretation |
|-----------|--------|----------------|
| SMA ratio | 25-day | Price relative to moving average |
| Bollinger %B | 20-day | Position within Bollinger Bands (>0.8 = overbought) |
| Momentum | 25-day | Price change ratio |
| PPO histogram | 12/26/9 | MACD-style trend strength |

---

## Background

Built on top of the Strategy Learner from GT CS 7646 Project 8, extended with:
- Live market data via Alpaca API (replacing the course's provided datasets)
- Gemini LLM research layer for qualitative/fundamental analysis
- Real paper trade execution via Alpaca brokerage API
