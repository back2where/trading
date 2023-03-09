import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

from datetime import datetime
import backtrader as bt


# Create a Stratey
class BandH(bt.Strategy):
    params = (
        ('startdate', pd.to_datetime('2013-01-02')),
        ('enddate', pd.to_datetime('2022-06-30'))
    )
    PortfolioValue = []
    RSI_daily = []
    SMA_daily = []
    MACD_daily = []
    OBV_daily = []
    CCI_daily = []

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


        # # Indicators for the plotting show
        # bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
        #                                     subplot=True)
        # bt.indicators.StochasticSlow(self.datas[0])
        # bt.indicators.MACDHisto(self.datas[0])
        self.RSI = bt.indicators.RSI(self.datas[0])
        # self.obv = bt.indicators.OBV(self.datas[0])
        self.MACD = bt.indicators.MACD(self.datas[0])
        self.CCI = bt.indicators.CCI(self.datas[0])
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
        self.CCI_daily.append([self.datas[0].datetime.date(0), self.CCI[0]])
        self.MACD_daily.append([self.datas[0].datetime.date(0), self.MACD[0]])
        self.RSI_daily.append([self.datas[0].datetime.date(0), self.RSI[0]])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.datas[0].datetime.date(0) == self.params.startdate:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.datas[0].datetime.date(0) == self.params.enddate:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(BandH)

    StartDate = '2013-01-01'
    EndDate = '2022-06-30'
    # Code = 'sh'
    # 载入数据
    # df = ts.get_k_data(Code, autype='qfq', start=StartDate, end=EndData)
    df = pd.read_excel(r'C:\Users\chen\OneDrive\科研\论文写作\强化学习\数据\CA\北京碳市场交易数据.xlsx')
    # df = pd.read_excel(r'C:\Users\chen\OneDrive\科研\论文写作\强化学习\数据\Iron\国内铁矿石连续.xlsx')
    df = df[(pd.to_datetime(StartDate) <= df['date']) & (df['date'] <= pd.to_datetime(EndDate))]
    df.index = pd.to_datetime(df.date)
    df = df[['open', 'high', 'low', 'close', 'volume']]  # 这些变量名要固定，之后的模块根据这些变量名进行策略操作

    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=df)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000)  # 初始资金10W

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=500)  # 铁矿石stake=100 , HBEA=500, oil=100

    # Set the commission
    cerebro.broker.setcommission(commission=0.005)  # 铁矿石"万1"， HBEA“千分5”， oil为"0"

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()

    # 输出每个交易日资产组合的价值
    PV_list = BandH.PortfolioValue
    CCI = BandH.CCI_daily
    MACD = BandH.MACD_daily
    RSI = BandH.RSI_daily

    PV_pd = pd.DataFrame(data=PV_list, columns=['date', 'value'])
    CCI_pd = pd.DataFrame(data=CCI, columns=['date', 'CCI'])
    MACD_pd = pd.DataFrame(data=MACD, columns=['date', 'MACD'])
    RSI_pd = pd.DataFrame(data=RSI, columns=['date', 'RSI'])

    tec_inx = pd.merge(left=CCI_pd, right=MACD_pd, on='date')
    tec_inx = pd.merge(left=tec_inx, right=RSI_pd, on='date')
    tec_inx.to_csv(r'C:\Users\chen\OneDrive\科研\论文写作\强化学习\数据\CA\\TecInx\BJEA_TecInx.csv')