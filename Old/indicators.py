import pandas as pd


def author():
    return 'atung9'

'''
5 technical indicators
1. Bollinger Bands %B
2. SMA (simple moving average)
3. Momentum
4. Stochastic Oscillator Indicator
5. Percentage Price Oscillator

'''

def calc_SMA(values, window):
    SMA = values.rolling(window=window).mean()
    prices_SMA = values/SMA - 1
    return SMA, prices_SMA

def calc_bollinger_bands(prices, window = 20):
    SMA, pricesSMA = calc_SMA(prices,window)
    std = prices.rolling(window=window).std()
    top_band = SMA + (2*std)
    btm_band = SMA - (2*std)

    bb_value = (prices - btm_band)/(top_band - btm_band) #bollinger band percentage
    return bb_value, top_band, btm_band

def calc_momentum(prices, N):
    momentum = (prices / prices.shift(N)) - 1
    return momentum

def calc_stochastic(high, low, close, adj_close, window = 14):
    #formula for the stochastic oscillator
    #k = (c-L14)/(H14 - L14) * 100 reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    multiplier = adj_close / close #calculating multiplier to discern adjusted costs
    adj_high = high * multiplier
    adj_low = low * multiplier

    low_low = adj_low.rolling(window=window).min()
    high_high = adj_high.rolling(window=window).max()

    k = ((adj_close - low_low) / (high_high - low_low)) * 100

    return k

def calc_EMA(values, window):
    return values.ewm(span=window).mean()

def calc_ppo(prices, window1 = 12, window2 = 26, sig_window = 9):
    #formula for ppo reference: https://www.investopedia.com/terms/p/ppo.asp
    EMA12 = calc_EMA(prices,window1)
    EMA26 = calc_EMA(prices, window2)

    ppo = ((EMA12 - EMA26) / EMA26) * 100
    signal_line = calc_EMA(ppo, sig_window)
    ppo_histo = ppo - signal_line
    return ppo_histo, ppo, signal_line

def test_code():
    pass

if __name__ == "__main__":
    test_code()