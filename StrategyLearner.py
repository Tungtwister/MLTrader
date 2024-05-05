""""""
import config

"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 

Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 

-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 

Student Name: Austin Tung (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: atung9 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903966860 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""

import datetime as dt
import random

import pandas as pd
import numpy as np
import indicators as ind
import RTLearner as rtl
import BagLearner as bgl

from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce

client = StockHistoricalDataClient(config.public,config.secret)



class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = 0

    def x_indicators(
            self,
            symbol='IBM',
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000
    ):
        lookback_window = 50
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        yearAgo = today - timedelta(days=365)
        lookback = yearAgo - timedelta(days=lookback_window)

        # Creating request object
        request_params = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame.Day,
            start=lookback,
            end=yesterday
        )

        # Retrieve daily bars for Bitcoin in a DataFrame and printing it
        btc_bars = client.get_stock_bars(request_params)

        # Convert to dataframe
        data_df = btc_bars.df
        data_df = data_df["close"]

        sma, sma_ratio = ind.calc_SMA(data_df, 25)
        bbp, top_band, btm_band = ind.calc_bollinger_bands(data_df, 20)
        momentum = ind.calc_momentum(data_df, 25)
        ppo_histo, ppo, signal_line = ind.calc_ppo(data_df)

        indicator_df = pd.concat([sma_ratio, bbp, momentum, ppo_histo], axis=1)
        indicator_df.columns = ['SMA', 'BBP', 'MOM', 'PPO']
        indicator_df = indicator_df.reset_index(level=[0])
        indicator_df = indicator_df.drop(columns="symbol")
        indicator_df.index =  indicator_df.index.tz_localize(None)
        indicator_df = indicator_df.loc[yearAgo:]
        print(indicator_df)
        return 1

    def y_train(self, indicators, trades, lookahead, symbol):
        ret = (trades.shift(-lookahead) / trades) - 1.0

        trades = ret.copy()
        trades.iloc[:, :] = 0
        for i in range(len(ret)):
            if ret.iloc[i, 0] > 0 + self.impact:
                trades.iloc[i, 0] = 1
            elif ret.iloc[i, 0] < 0 - self.impact:
                trades.iloc[i, 0] = -1
        return trades

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        x_train, trades = self.x_indicators(symbol, sd, ed, sv)
        x_train = x_train.to_numpy()  # issues passing in dataframe for rtlearner reference: https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
        y_train = self.y_train(x_train, trades, 5, symbol)
        y_train = y_train.to_numpy()
        x_train = x_train[
                  1:]  # cut off first trading day as it should be nan because while we can prime our indicators we don't trade using information outside the in-sample
        y_train = y_train[1:]
        self.learner = bgl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 5}, bags=100, boost=False,
                                      verbose=False)
        self.learner.add_evidence(x_train, y_train)

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        x_test, trades = self.x_indicators(symbol, sd, ed, sv)
        x_test = x_test.to_numpy()
        trades.values[:, :] = 0  # set them all to nothing

        pred_y = self.learner.query(x_test)
        totShares = 0  # code adapted from project 6
        for i in range(1, len(trades) - 1):
            if (pred_y[0, i] == 1) and totShares < 1000:
                if (totShares == -1000):
                    totShares += 2000
                    trades.values[i, 0] = 2000
                else:
                    totShares += 1000
                    trades.values[i, 0] = 1000
            elif (pred_y[0, i] == -1) and totShares > -1000:
                if (totShares == 1000):
                    totShares -= 2000
                    trades.values[i, 0] = -2000
                else:
                    totShares -= 1000
                    trades.values[i, 0] = -1000

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(trades)
        return trades

    def author():
        return "atung9"


if __name__ == "__main__":
    learner = StrategyLearner()
    test_ind = learner.x_indicators(symbol='IBM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),sv=10000)
    print(test_ind)