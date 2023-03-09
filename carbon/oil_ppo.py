import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.font_manager as fm
import time

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DDPG

from oilenv.oiltrain import StockEnvTrain
from oilenv.oilval import StockEnvValidation
from oilenv.oiltrade import StockEnvTrade

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

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

def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)

#def train_PPO(env_train, timesteps=35000):
def train_PPO(env_train, timesteps=10000):
    """PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 6)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_DDPG(env_train, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']#columns属性返回给定Dataframe的列标签。
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             (df_total_value['daily_return'].std()+ float("1e-8"))
    return sharpe

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   # turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   # turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])

    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    return last_state


if __name__ == '__main__':
    #data = pd.read_csv('./data/brent.csv')
    data = pd.read_csv('./data/brent_vmfTech2.csv')
    # data = pd.read_csv('./data/brent_vmfTech.csv')
    unique_trade_date = data[(data.trade_date >= '2018-10-10') & (data.trade_date <= '2022-06-30')].trade_date.unique()
    df = data
    print("============Start Ensemble Strategy============")
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []

    rebalance_window = 60
    validation_window = 60
    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date),
                   rebalance_window):  # range(start, stop, step)
        print("=======================*****************************=====================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start='2015-01-02', end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation, iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======training from: ", '2015-01-02', "to ", unique_trade_date[i - rebalance_window - validation_window])
        print("======PPO Training========")
        #model_ppo = train_PPO(env_train, timesteps=100000)
        model_ppo = train_PPO(env_train, timesteps=20000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)
        model_ensemble = model_ppo
        model_use.append('PPO')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        print("Used Model: ", model_ensemble)
        trade = data_split(df, start=unique_trade_date[i - rebalance_window], end=unique_trade_date[i])

        env_trade = DummyVecEnv([lambda: StockEnvTrade(trade)])
        obs_trade = env_trade.reset()
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ppo",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             initial=initial)

        print(last_state_ensemble)


    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
