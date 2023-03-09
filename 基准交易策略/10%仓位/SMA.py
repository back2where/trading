# 参考：https://zhuanlan.zhihu.com/p/140425363
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

from datetime import datetime
import backtrader as bt
import backtrader.analyzers as btanalyzers

# Create a Stratey
class SMA(bt.Strategy):
    params = (
        ('maperiod', 20),
    )
    PortfolioValue = []

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

        # # Indicators for the plotting show
        # bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
        #                                     subplot=True)
        # bt.indicators.StochasticSlow(self.datas[0])
        # bt.indicators.MACDHisto(self.datas[0])
        # rsi = bt.indicators.RSI(self.datas[0])
        # bt.indicators.SmoothedMovingAverage(rsi, period=10)
        # bt.indicators.ATR(self.datas[0], plot=False)

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

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        self.PortfolioValue.append([self.datas[0].datetime.date(0), self.broker.getvalue()])  # 记录每个交易日资产组合的价值

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



if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(SMA)

    StartDate = '2018-12-04'
    EndDate = '2022-06-30'
    # Code = 'sh'
    #获取数据
    # df = ts.get_k_data(Code, autype='qfq', start=StartDate, end=EndData)
    df = pd.read_excel(r'C:\Users\chen\OneDrive\科研\论文写作\强化学习\数据\CA\深圳碳市场交易数据.xlsx')
    df = df[(pd.to_datetime(StartDate) <= df['date']) & (df['date'] <= pd.to_datetime(EndDate))]
    df.index = pd.to_datetime(df.date)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=df)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=500)  # 铁矿石stake=100 , HBEA=500, Oil=100

    # Set the commission
    cerebro.broker.setcommission(commission=0.005)  # 铁矿石"万1"， HBEA“千分5”， oil为"0"

    # Analyzer-------------------
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='shape')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    back = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()

    # 输出每个交易日资产组合的价值
    PV_list = SMA.PortfolioValue
    PV_pd = pd.DataFrame(data=PV_list, columns=['date', 'value'])
    PV_pd.to_csv(r'C:\Users\chen\OneDrive\科研\论文写作\强化学习\实证结果\CA\SMA.csv')

    back[0].analyzers.returns.get_analysis()
    back[0].analyzers.shape.get_analysis()
    back[0].analyzers.drawdown.get_analysis()
