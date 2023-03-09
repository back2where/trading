# 参考: https://zhuanlan.zhihu.com/p/122183963
import pandas as pd
from datetime import datetime
import backtrader as bt
import matplotlib.pyplot as plt

# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()

# 设计交易策略
class BuyHold(bt.Strategy)：

    def log(self,)





# 使用tushare旧版接口获取数据
import tushare as ts


def get_data(code,start='2010-01-01',end='2020-03-31'):
    df=ts.get_k_data(code,autype='qfq',start=start,end=end)
    df.index=pd.to_datetime(df.date)
    df['openinterest']=0
    df=df[['open','high','low','close','volume','openinterest']]
    df=df.sort_index()
    return df



# 回测期间
dataframe=get_data('600000')
start = datetime(2010, 3, 31)
end = datetime(2020, 3, 31)
# 加载数据
data = bt.feeds.PandasData(dataname=dataframe, fromdate=start, todate=end)

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Create a Data Feed
    dataframe = get_data('600000')
    data = bt.feeds.PandasData(dataname=dataframe, fromdate=start, todate=end)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
