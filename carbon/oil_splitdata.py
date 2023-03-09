import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from RLIRONORE.O.test.Brent_WTI_all_frequency import AllDecompositionEnv
from RLIRONORE.O.test.oil_high_frequency import WithhighfrequencyEnv
from RLIRONORE.O.test.oil_without import WithoutDecompositionEnv
from RLIRONORE.O.test.oil_high_low import WithhighlowfrequencyEnv
import matplotlib.pyplot as plt
# mpl.use('TkAgg')
import matplotlib.font_manager as fm
# import seaborn as  sns
# sns.set() # 因为sns.set()一般不用改，可以在导入模块时顺便设置好
font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('./data/WTIoil_price.xlsx', index_col=0)
    data_xls.to_csv('./data/WTIoil_price.csv', encoding='utf-8')

def csv_to_multi():
    df = pd.read_csv('oildata.csv')
    dfn=df[892:]#取前一万行
    dfn.to_csv('892.csv',index=False)

def Without(oil_file1,oil_file2):
    df_train = oil_file1
    # df_train = df_train.sort_values('trade_date')

    env = DummyVecEnv([lambda: WithoutDecompositionEnv(df_train)])
    # model = A2C(MlpPolicy, env, verbose=1)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000)
    df_test = oil_file2
    env_test = DummyVecEnv([lambda: WithoutDecompositionEnv(df_test)])
    # model.save("./ppo_price")
    # del model
    # model = PPO2.load("./ppo_price",env)
    # model.learn(100)

    day_profits = []
    obs = env_test.reset()
    for i in range(len(df_test)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
    return day_profits

def With_All(oil_file1,oil_file2):
    df_train = oil_file1
    # df_train = df_train.sort_values('trade_date')

    env = DummyVecEnv([lambda: AllDecompositionEnv(df_train)])
    # model = A2C(MlpPolicy, env, verbose=1)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000)
    df_test = oil_file2
    env_test = DummyVecEnv([lambda: AllDecompositionEnv(df_test)])
    # model.save("./ppo_price")
    # del model
    # model = PPO2.load("./ppo_price",env)
    # model.learn(100)

    day_profits = []
    obs = env_test.reset()
    for i in range(len(df_test)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
    return day_profits

def With_High(oil_file1,oil_file2):
    df_train = oil_file1
    # df_train = df_train.sort_values('trade_date')

    env = DummyVecEnv([lambda: WithhighfrequencyEnv(df_train)])
    # model = A2C(MlpPolicy,env,verbose=1)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000)
    df_test = oil_file2
    env_test = DummyVecEnv([lambda: WithhighfrequencyEnv(df_test)])
    # model.save("./ppo_price")
    # del model
    # model = PPO2.load("./ppo_price",env)
    # model.learn(100)

    day_profits = []
    obs = env_test.reset()
    for i in range(len(df_test)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
    return day_profits

def With_High_Low(oil_file1,oil_file2):
    df_train = oil_file1
    # df_train = df_train.sort_values('trade_date')

    env = DummyVecEnv([lambda: WithhighlowfrequencyEnv(df_train)])
    # model = A2C(MlpPolicy,env,verbose=1)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000)
    df_test = oil_file2
    env_test = DummyVecEnv([lambda: WithhighlowfrequencyEnv(df_test)])
    # model.save("./ppo_price")
    # del model
    # model = PPO2.load("./ppo_price",env)
    # model.learn(100)

    day_profits = []
    obs = env_test.reset()
    for i in range(len(df_test)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
    return day_profits

# def data_split(df,start,end):
#     """
#     split the dataset into training or testing using date
#     :param data: (df) pandas dataframe, start, end
#     :return: (df) pandas dataframe
#     """
#     data = df[(df.trade_date >= start) & (df.trade_date < end)]
#     data=data.sort_values(['trade_date'],ignore_index=True)
#     #data  = data[final_columns]
#     data.index = data.trade_date.factorize()[0]
#     return data
def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.trade_date >= start) & (df.trade_date < end)]
    data=data.sort_values(['trade_date'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.trade_date.factorize()[0]
    return data

if __name__ == '__main__':
    # xlsx_to_csv_pd()
    # data = pd.read_csv('./data/brent.csv')
    # data = pd.read_csv('./data/wti.csv')
    data = pd.read_csv('./data/brent_vmfTech.csv')
    # data = pd.read_csv('./data/wti_vmf.csv')
    # df= data.iloc[:,2:11]
    # print(df)
    # data['imf_sum'] = data.iloc[:, 2:11].sum(axis=1)
    print(data)
    # Bo_file1 = data_split(df3, '2015-01-02', '2019-01-02')
    # Bo_file2 = data_split(df3, '2019-01-02', '2022-06-30')
    # print('Brentoil_vmf:',Bo_file2.shape)

    unique_trade_date = data[(data.trade_date >= '2018-10-10') & (data.trade_date <= '2022-06-30')].trade_date.unique()
    print(unique_trade_date)
    df = data

    rebalance_window = 60
    validation_window = 60

    for i in range(rebalance_window + validation_window, len(unique_trade_date),
                   rebalance_window):  # range(start, stop, step)
        print("=======================*****************************=====================")
        print("===========Training=================")
        print("======training from: ", '2015-01-02', "to ", unique_trade_date[i - rebalance_window - validation_window])
        # print("training period:",unique_trade_date[i - rebalance_window - validation_window])
        print("===========Validate=================")
        print("======Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        # print("validation period_{}_{}:".format(validation, len(validation)))
        print("===========Trading=================")
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        trade = data_split(df, start=unique_trade_date[i - rebalance_window], end=unique_trade_date[i])
        print("=======================*****************************=====================")
