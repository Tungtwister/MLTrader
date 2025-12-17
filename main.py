from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import config

# 1. Setup the Historical Data Client (Use your API keys)
stock_historical_data_client = StockHistoricalDataClient(config.public, config.secret)

# 2. Define the lookback period (2 years)
# We use New York time to match market hours
now = datetime.now(ZoneInfo("America/New_York"))
yesterday = now - timedelta(days=1)
start_date = now - timedelta(days=365 * 2) 

# 3. Create the Request
# We change the timeframe to 'Day' because 2 years of minute/hour data is very large
req = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start=start_date,
    end=yesterday
)

# 4. Fetch the data and convert to a Pandas DataFrame
bars = stock_historical_data_client.get_stock_bars(req)
df = bars.df

# Display the first few rows
print(df.head())
print(f"Total rows: {len(df)}")