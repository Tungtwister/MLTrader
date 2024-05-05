from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest



trading_client = TradingClient('PKDDA61X3SOWD31DTDXF', 'FTlc4NHJ6Fp36BjFgdY8lElRdyeTjhgFOwEcdRD0', paper=True)

# preparing market order
def prepareOrder (symbol, qty, side, tif):
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=tif
                        )
    return market_order_data

# Market order
def submitOrder (market_order_data):
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                   )
    return market_order