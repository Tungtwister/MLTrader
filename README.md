# MLTrader

A hybrid ML + LLM paper trading bot built as a continuation of Georgia Tech's CS 7646: Machine Learning for Trading.

The system combines a Bagged Random Tree technical model with a Gemini-powered research agent to generate trade signals, then executes them via the Alpaca paper trading API.

The ML model is the primary driver. The LLM acts as a selective veto — it only blocks a trade when it strongly disagrees (opposite direction, confidence >= 0.70). This approach consistently outperformed both the original "both must agree" logic and a naive buy-and-hold baseline in backtesting.

---

## How It Works

```
python main.py SPY
       |
       |-- StrategyLearner (Bagged Random Trees)
       |     Trains on 2 years of daily OHLCV data
       |     Features: SMA ratio, Bollinger %B, Momentum, PPO histogram,
       |               ATR% (volatility), Volume Ratio
       |     20-day lookahead labels (reduces noise vs original 5-day)
       |     Long-bias mode for ETFs: start invested, only exit on sell signal
       |     Signals: +1000 (buy), -1000 (sell), 0 (hold)
       |
       |-- ResearchAgent (Gemini 2.5-flash-lite via REST API)
       |     Fetches live news from Alpaca News API
       |     Fetches revenue / net income / EPS from SEC EDGAR
       |     Returns: { signal, confidence, reasoning }
       |
       |-- Signal Combiner (ML drives, LLM vetoes)
       |     ML signal is executed unless LLM strongly disagrees
       |     Veto requires: opposite direction AND confidence >= 0.70
       |     ETFs (SPY, QQQ, etc.) are never shorted
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
|-- indicators.py        # Technical indicators: SMA, BBP, Momentum, PPO, ATR%, Volume Ratio
|-- RTLearner.py         # Random Tree decision tree (classification)
|-- BagLearner.py        # Bootstrap aggregating (100 bags)
|-- placeOrder.py        # Alpaca paper trading wrapper
|-- analysis.ipynb       # Strategy evolution notebook — lessons learned + backtest results
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
[StrategyLearner] Trained on 249 samples (SPY) | labels: {1.0: 142, 0.0: 68, -1.0: 39}
[SPY] Signaling : 2025-05-17 -> 2026-05-16 (long_bias=True)

[SPY] Technical signal : +1000
[SPY] Current position : +0 shares
[SPY] Latest indicators: {'SMA': 0.028, 'BBP': 0.782, 'MOM': 0.088, 'PPO': 0.065, 'ATR_PCT': 0.011, 'VOL_RATIO': 1.14}

[SPY] Running LLM research...
[SPY] LLM signal     : BUY
[SPY] LLM confidence : 0.72
[SPY] LLM reasoning  : Momentum and volume confirm trend; recent news broadly positive for index.

[SPY] Final decision : BUY
[SPY] Opened long  -> order abc123
```

---

## Signal Logic

The ML model generates the primary signal. The LLM can only veto — it cannot independently trigger a trade.

| ML Signal | LLM Signal | LLM Confidence | Action |
|-----------|------------|----------------|--------|
| BUY | buy or hold | any | Execute buy |
| BUY | sell | >= 0.70 | Hold (vetoed) |
| BUY | sell | < 0.70 | Execute buy |
| SELL | sell or hold | any | Execute sell |
| SELL | buy | >= 0.70 | Hold (vetoed) |
| SELL | buy | < 0.70 | Execute sell |
| HOLD | any | any | Hold |
| SELL on ETF | — | — | Hold (no shorting) |

**ETFs that are never shorted:** SPY, QQQ, IWM, DIA, VTI, VOO, IVV, GLD

---

## Technical Indicators

| Indicator | Window | What it captures |
|-----------|--------|-----------------|
| SMA ratio | 25-day | Price deviation from moving average — trend direction |
| Bollinger %B | 20-day | Position within Bollinger Bands — overbought/oversold |
| Momentum | 25-day | Rate of price change — trend strength |
| PPO histogram | 12/26/9 | MACD-style divergence between short/long EMAs |
| ATR% | 14-day | Average True Range as % of price — normalised volatility |
| Volume Ratio | 20-day | Today's volume vs rolling average — conviction behind moves |

ATR% and Volume Ratio were added after backtesting showed the original 4-feature model produced mostly HOLD signals. The expanded feature set improved label coverage and trade frequency.

---

## Backtest Results

Tested out-of-sample (train on prior 2 years, test on most recent year):

| Symbol | Old Strategy | New Strategy | Buy & Hold |
|--------|-------------|--------------|------------|
| SPY | -4.7% | +8.8% | +24.7% |
| NVDA | -14.6% | +15.6% | +42.1% |

The new strategy consistently beats the old one and captures a meaningful portion of the underlying trend. Full analysis with charts in [`analysis.ipynb`](analysis.ipynb).

---

## Background

Built on top of the Strategy Learner from GT CS 7646 Project 8, extended with:
- Live market data via Alpaca API (replacing the course's provided datasets)
- Two new technical indicators (ATR%, Volume Ratio) for better signal coverage
- 20-day lookahead labels to reduce short-term noise in training
- Long-bias mode for ETFs — stays invested, only exits on strong sell signals
- LLM-as-veto architecture: Gemini provides qualitative/fundamental research and blocks trades only when it strongly disagrees
- Real paper trade execution via Alpaca brokerage API
