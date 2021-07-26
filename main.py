"""
Plan for this file.
    1. Create portfolio object from universe of presaved stocks
    2. Create agent objects
    3. Train agents.
    4. Test agent performance
"""

# from agents import *
from environments import *
from storage import *

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
matplotlib.use('Agg')
import datetime
from datetime import datetime as dt

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_portfolio import StockPortfolioEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts

import pyfolio
from pyfolio import timeseries

def get_snp500_stock_arguments():
    with open("data/snp500_stocks.txt", "r") as file:
        table = [stock.split("\t") for stock in file.read().split("\n")]
        headers = table[0]
        stock_arguments = table[1:]
        return stock_arguments

def refresh_saved_stocks_technical_only():
    stock_arguments = get_snp500_stock_arguments()
    for i in range(len(stock_arguments)):
        s = Stock(*stock_arguments[i])
        s.extract_and_calculate_technical()
        save_stock(s, "data/saved_stocks_technical_only")
        print(s)

def get_average_performance(agent, number, episodes, save_fig=True):
    returns_over_market = [[] for _ in range(episodes)]
    returns_over_stock = [[] for _ in range(episodes)]
    for _ in range(number):
        a = agent()
        a.train(episodes=episodes)
        for e in range(episodes):
            returns_over_market[e].append(a.return_over_market[e])
            returns_over_stock[e].append(a.return_over_stock[e])
    average_return_over_market = [sum(vs)/len(vs) for vs in returns_over_market]
    average_return_over_stock = [sum(vs)/len(vs) for vs in returns_over_stock]
    # Plot perfomance
    if save_fig:
        plt.plot(average_return_over_market, label="Over Market")
        plt.plot(average_return_over_stock, label="Over Stock")
        title = f"{number} Agents over {episodes} episodes"
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Relative return")
        plt.legend()
        plt.savefig(f"data/figs/{title}")
        plt.clf()
    return average_return_over_market, average_return_over_stock

def test_accuracy(environments, date_pairs):
    def all_agents_trained(environment):
        agents_dict = dict()
        # A2C
        agent = DRLAgent(env = env_train)
        A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
        model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
        agents_dict['a2c'] = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=50000)
        # PPO
        agent = DRLAgent(env = env_train)
        PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
        agents_dict['ppo'] = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
        # DDPG
        agent = DRLAgent(env = env_train)
        DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
        model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
        agents_dict['ddpg'] = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=50000)
        # SAC
        agent = DRLAgent(env = env_train)
        SAC_PARAMS = {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0003,  "learning_starts": 100, "ent_coef": "auto_0.1"}
        model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
        agents_dict['sac'] = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=50000)
        # TD3
        agent = DRLAgent(env = env_train)
        TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
        agents_dict['td3'] = agent.get_model("td3",model_kwargs = TD3_PARAMS)
        return agents_dict

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)
    # GETTING DATAFRAME
    df = YahooDownloader(start_date = '2008-01-01',
                        end_date = '2021-07-01',
                        ticker_list = config.DOW_30_TICKER).fetch_data()
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=False,
                        user_defined_feature = False)
    df = fe.preprocess_data(df)
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    cov_list = []
    return_list = []
    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        covs = return_lookback.cov().values 
        cov_list.append(covs)

    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    # TESTING
    e_count = 0
    for e in environments:
        e_count += 1
        d_count = 0
        for dates in date_pairs:
            d_count += 1
            # CREATE ENVIRONMENT    
            train = data_split(df, dates[0][0], dates[0][1])
            stock_dimension = len(train.tic.unique())
            state_space = stock_dimension
            env_kwargs = {
                "hmax": 100, 
                "initial_amount": 1000000, 
                "transaction_cost_pct": 0.001, 
                "state_space": state_space, 
                "stock_dim": stock_dimension, 
                "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
                "action_space": stock_dimension, 
                "reward_scaling": 1e-4    
            }
            e_train_gym = e(df = train, **env_kwargs)
            env_train, obs = e_train_gym.get_sb_env()

            # CREATE AND TRAIN AGENTS
            agents_dict = all_agents_trained(env_train)

            for a in agents_dict:

                # TEST AGENT PERFORMANCE
                trade = data_split(df, dates[1][0], dates[1][1])
                e_trade_gym = e(df = trade, **env_kwargs)
                df_daily_return, df_actions = DRLAgent.DRL_prediction(model=agents_dict[a], environment=e_trade_gym)
                DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
                perf_func = timeseries.perf_stats 
                perf_stats_all = perf_func(returns=DRL_strat, factor_returns=DRL_strat, positions=None, transactions=None, turnover_denom="AGB")
                baseline_df = get_baseline(ticker="^DJI", start=dates[1][0], end=dates[1][1])
                print("==============Get Baseline Stats===========")
                stats = backtest_stats(baseline_df, value_col_name = 'close')
                print("==============DRL Strategy Stats===========")
                print(perf_stats_all)

                # CREATE A RECORD
                with open(f"Agent-{a}_Env-{e_count}_dates-{d_count}.txt", "w") as file:
                    file.write(f"{e}\n{e.__class__.__name__}\n{str(dates)}\n===Baseline Stats===\n{stats}\n===DRL State===\n{perf_stats_all}")



def main():
    # training_dates = ('2014-01-01','2020-07-01')
    # testing_dates = ('2020-07-01', '2021-07-01')
    date_pairs = [(('2014-01-01','2020-07-01'), ('2020-07-01', '2021-07-01')), 
                    ((('2013-01-01','2019-07-01'), ('2019-07-01', '2020-07-01'))),
                    ((('2012-01-01','2018-07-01'), ('2018-07-01', '2019-07-01'))),
                    ((('2011-01-01','2017-07-01'), ('2017-07-01', '2018-07-01'))),
                    ((('2010-01-01','2016-07-01'), ('2016-07-01', '2017-07-01')))]
    test_accuracy([StockPortfolioEnv, TIOnlyStockPortfolioEnv], date_pairs)

    # df = pd.read_csv("data/df.csv.zip", compression="zip")

    # # Create Environment

    # train = data_split(df, '2009-01-01','2020-07-01')

    # stock_dimension = len(train.tic.unique())
    # state_space = stock_dimension
    # env_kwargs = {
    #     "hmax": 100, 
    #     "initial_amount": 1000000, 
    #     "transaction_cost_pct": 0.001, 
    #     "state_space": state_space, 
    #     "stock_dim": stock_dimension, 
    #     "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    #     "action_space": stock_dimension, 
    #     "reward_scaling": 1e-4
        
    # }

    # e_train_gym = TIOnlyStockPortfolioEnv(df = train, **env_kwargs)
    # env_train, obs = e_train_gym.get_sb_env()


    # # Train Agent

    # agent = DRLAgent(env = env_train)
    # DDPG_PARAMS = {"batch_size": 128, "buffer_size": 5000, "learning_rate": 0.001}
    # model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
    # trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=50)

    # # Test Agent

    # trade = data_split(df,'2020-07-01', '2021-07-01')
    # e_trade_gym = TIOnlyStockPortfolioEnv(df = trade, **env_kwargs)

    # df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)

    # DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    # perf_func = timeseries.perf_stats 
    # perf_stats_all = perf_func(returns=DRL_strat, factor_returns=DRL_strat, positions=None, transactions=None, turnover_denom="AGB")

    # baseline_df = get_baseline(ticker="^DJI", start = df_daily_return.loc[0,'date'], end = df_daily_return.loc[len(df_daily_return)-1,'date'])
    
    # print("==============Get Baseline Stats===========")
    # stats = backtest_stats(baseline_df, value_col_name = 'close')


    # print("==============DRL Strategy Stats===========")
    # print(perf_stats_all)


    # with open(f"text.txt", "w") as file:
    #             file.write(f"===Baseline Stats===\n{stats}\n===DRL State===\n{perf_stats_all}")

if __name__ == "__main__":
    main()


