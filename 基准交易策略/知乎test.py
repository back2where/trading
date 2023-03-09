# 参考：https://zhuanlan.zhihu.com/p/140425363
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

from datetime import datetime
import backtrader as bt


class SMA20_Strategy(bt.Strategy):
    params = (
        ('maperiod', 20),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


# def main(code,start,end='',startcash=10000,qts=500,com=0.001):
#     #创建主控制器
#     cerebro = bt.Cerebro()
#     #导入策略参数寻优
#     cerebro.optstrategy(MyStrategy, maperiod=range(3, 31))
#     #获取数据
#     df=ts.get_k_data(code,autype='qfq',start=start,end=end)
#     df.index=pd.to_datetime(df.date)
#     df=df[['open','high','low','close','volume']]
#     #将数据加载至回测系统
#     data = bt.feeds.PandasData(dataname=df)
#     cerebro.adddata(data)
#     #broker设置资金、手续费
#     cerebro.broker.setcash(startcash)
#     cerebro.broker.setcommission(commission=com)
#     #设置买入设置，策略，数量
#     cerebro.addsizer(bt.sizers.FixedSize, stake=qts)
#     print('期初总资金: %.2f' %
#     cerebro.broker.getvalue())
#     cerebro.run(maxcpus=1)
#     print('期末总资金: %.2f' % cerebro.broker.getvalue())

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(SMA20_Strategy)

    StartDate = '2017-01-01'
    EndData = '2021-08-31'
    Code = 'sh'
    #获取数据
    df = ts.get_k_data(Code, autype='qfq', start=StartDate, end=EndData)
    df.index = pd.to_datetime(df.date)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=df)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000000)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())