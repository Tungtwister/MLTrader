from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

trading_client = TradingClient('PKDDA61X3SOWD31DTDXF', 'FTlc4NHJ6Fp36BjFgdY8lElRdyeTjhgFOwEcdRD0')

# Get our account information.
account = trading_client.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

# Get our account information.
account = trading_client.get_account()

# Check our current balance vs. our balance at the last market close
balance_change = float(account.equity) - float(account.last_equity)
print(f'Today\'s portfolio balance change: ${balance_change}')

search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

assets = trading_client.get_all_assets(search_params)