import pandas as pd
import placeOrder as po
import indicators as ind
import config
import StrategyLearner as sl
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce

client = StockHistoricalDataClient(config.public,config.secret)

today = datetime.now().date()
yesterday = today - timedelta(days=1)
yearAgo = today - timedelta(days=365)

# Creating request object
request_params = StockBarsRequest(
  symbol_or_symbols=["SPY"],
  timeframe=TimeFrame.Day,
  start=yearAgo,
  end=yesterday
)

# Retrieve daily bars for Bitcoin in a DataFrame and printing it
btc_bars = client.get_stock_bars(request_params)

# Convert to dataframe
data_df = btc_bars.df

sma, price_sma = ind.calc_SMA(data_df,15)
bb, top, bot = ind.calc_bollinger_bands(data_df)

print(data_df)