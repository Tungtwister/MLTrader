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
# print(df.tail())
# print(f"Total rows: {len(df)}")

# 1. Define your "Profit Threshold"
# 0.001 means 0.1%. If the stock goes up less than this, we stay in cash (0).
THRESHOLD = 0.001 

# 2. Calculate Tomorrow's Return
# We shift the 'close' column up by 1 (-1) to align tomorrow's price with today's row.
df['next_close'] = df['close'].shift(-1)
df['next_return'] = (df['next_close'] - df['close']) / df['close']

# 3. Create the Binary Target
# 1 = Profitable Buy, 0 = Hold/Sell
df['Target'] = (df['next_return'] > THRESHOLD).astype(int)

# 4. Drop the last row (it has NaN because there is no 'tomorrow' for it)
df.dropna(subset=['next_close'], inplace=True)

# --- Verification ---
print("Target Distribution:")
print(df['Target'].value_counts(normalize=True))

print("\nFirst 5 rows with Targets:")
print(df.tail(20))