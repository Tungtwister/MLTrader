from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import indicators as ind
import RTLearner as rtl
import BagLearner as bgl

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

_client = StockHistoricalDataClient(config.public, config.secret)

# Extra calendar days fetched before sd to warm up rolling indicators
_WARMUP_DAYS = 100

# Feature columns used for training / inference
_FEATURES = ["SMA", "BBP", "MOM", "PPO", "ATR_PCT", "VOL_RATIO"]


def _to_naive(dt_val):
    """Strip timezone from a datetime so it can index a tz-naive DatetimeIndex."""
    if isinstance(dt_val, datetime) and dt_val.tzinfo is not None:
        return dt_val.replace(tzinfo=None)
    return dt_val


class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_indicators(self, symbol, sd, ed):
        """
        Fetch daily bars for symbol over [sd, ed] (with a warmup prefix),
        compute all six technical indicators, and return:
            indicator_df  — DataFrame(SMA, BBP, MOM, PPO, ATR_PCT, VOL_RATIO)
            prices_df     — DataFrame(close)
        Both DataFrames are sliced to exactly [sd, ed] after warmup.
        """
        fetch_start = _to_naive(sd) - timedelta(days=_WARMUP_DAYS)
        fetch_end   = _to_naive(ed)

        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=fetch_start,
            end=fetch_end,
        )
        bars = _client.get_stock_bars(req)
        df   = bars.df

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        _, sma_ratio = ind.calc_SMA(close, 25)
        bbp, _, _    = ind.calc_bollinger_bands(close, 20)
        momentum     = ind.calc_momentum(close, 25)
        ppo_histo, _, _ = ind.calc_ppo(close)
        atr_pct      = ind.calc_atr_pct(high, low, close, 14)
        vol_ratio    = ind.calc_volume_ratio(volume, 20)

        indicator_df = pd.concat(
            [sma_ratio, bbp, momentum, ppo_histo, atr_pct, vol_ratio], axis=1
        )
        indicator_df.columns = _FEATURES
        indicator_df = indicator_df.reset_index(level="symbol", drop=True)
        indicator_df.index = indicator_df.index.tz_localize(None)

        prices_df = close.reset_index(level="symbol", drop=True).to_frame("close")
        prices_df.index = prices_df.index.tz_localize(None)

        # Slice to the requested window (drop warmup rows)
        sd_n = _to_naive(sd)
        ed_n = _to_naive(ed)
        indicator_df = indicator_df.loc[sd_n:ed_n]
        prices_df    = prices_df.loc[sd_n:ed_n]

        return indicator_df, prices_df

    def _make_labels(self, prices_df, lookahead=20):
        """
        Classification labels from future returns over `lookahead` trading days:
          +1  -> buy  (expected return > impact)
          -1  -> sell (expected return < -impact)
           0  -> hold / uncertain

        Using a 20-day lookahead (vs the original 5) significantly reduces label
        noise because short-term daily moves are mostly random.
        """
        ret = prices_df["close"].shift(-lookahead) / prices_df["close"] - 1.0
        labels = pd.Series(0.0, index=prices_df.index)
        labels[ret >  self.impact] =  1.0
        labels[ret < -self.impact] = -1.0
        return labels

    # ── Public API ────────────────────────────────────────────────────────────

    def add_evidence(self, symbol="SPY", sd=None, ed=None, sv=10000, lookahead=20):
        """Train the strategy learner on historical data for symbol over [sd, ed]."""
        now = datetime.now(ZoneInfo("America/New_York"))
        if sd is None:
            sd = now - timedelta(days=730)
        if ed is None:
            ed = now - timedelta(days=365)

        indicator_df, prices_df = self._fetch_indicators(symbol, sd, ed)
        labels = self._make_labels(prices_df, lookahead=lookahead)

        combined = indicator_df.copy()
        combined["label"] = labels
        combined.dropna(inplace=True)

        x_train = combined[_FEATURES].to_numpy()
        y_train = combined["label"].to_numpy()

        self.learner = bgl.BagLearner(
            learner=rtl.RTLearner,
            kwargs={"leaf_size": 5},
            bags=100,
            boost=False,
            verbose=False,
        )
        self.learner.add_evidence(x_train, y_train)

        if self.verbose:
            label_counts = pd.Series(y_train).value_counts().to_dict()
            print(f"[StrategyLearner] Trained on {len(x_train)} samples ({symbol}) "
                  f"| labels: {label_counts}")

    def get_latest_indicators(self, symbol: str, lookback_days: int = 60) -> dict:
        """Return the most recent computed indicator values as a plain dict."""
        now = datetime.now(ZoneInfo("America/New_York"))
        sd = now - timedelta(days=lookback_days)
        ed = now - timedelta(days=1)
        indicator_df, _ = self._fetch_indicators(symbol, sd, ed)
        indicator_df.dropna(inplace=True)
        if indicator_df.empty:
            return {}
        return indicator_df.iloc[-1].to_dict()

    def testPolicy(self, symbol="SPY", sd=None, ed=None, sv=10000,
                   long_bias=False, lookahead=20):
        """
        Run the trained policy over [sd, ed].

        long_bias=True  (recommended for ETFs / upward-trending assets):
            Start fully invested. Only exit on a sell signal, re-enter on buy.
            Captures the natural upward drift instead of sitting in cash.

        long_bias=False (original behaviour):
            Start flat. Enter on buy signal, exit/short on sell signal.

        Returns a DataFrame of trade actions indexed by date:
            +1000  buy 1000 shares
            -1000  sell / close 1000 shares
                0  no change
        """
        if self.learner is None:
            raise RuntimeError("Call add_evidence() before testPolicy().")

        now = datetime.now(ZoneInfo("America/New_York"))
        if sd is None:
            sd = now - timedelta(days=365)
        if ed is None:
            ed = now - timedelta(days=1)

        indicator_df, _ = self._fetch_indicators(symbol, sd, ed)
        indicator_df.dropna(inplace=True)

        x_test = indicator_df.to_numpy()
        pred_y = self.learner.query(x_test)  # shape (1, n)

        trades     = pd.DataFrame(0.0, index=indicator_df.index, columns=["shares"])
        tot_shares = 1000 if long_bias else 0

        for i in range(len(indicator_df)):
            signal = pred_y[0, i]

            if long_bias:
                # Only two states: long (1000) or flat (0) — no shorting
                if signal == -1.0 and tot_shares > 0:
                    trades.iloc[i, 0] = -1000
                    tot_shares = 0
                elif signal == 1.0 and tot_shares == 0:
                    trades.iloc[i, 0] = 1000
                    tot_shares = 1000
            else:
                # Original three-state behaviour: long / flat / short
                if signal == 1.0 and tot_shares < 1000:
                    delta = 2000 if tot_shares == -1000 else 1000
                    trades.iloc[i, 0] = delta
                    tot_shares += delta
                elif signal == -1.0 and tot_shares > -1000:
                    delta = 2000 if tot_shares == 1000 else 1000
                    trades.iloc[i, 0] = -delta
                    tot_shares -= delta

        return trades
