import config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest

_trading_client = TradingClient(config.public, config.secret, paper=True)


def prepareOrder(symbol, qty, side, time_in_force):
    return MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=time_in_force,
    )


def submitOrder(market_order_data):
    return _trading_client.submit_order(order_data=market_order_data)
