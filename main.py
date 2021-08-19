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
from constants import *

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

def snp_stocks_basic():
    for i in range(0, len(SNP_500_TOP_100)):
        s = Stock(*SNP_500_TOP_100[i])
        s.extract_and_calculate_basic(verbose=False)
        s.calculate_cheat_values()
        save_stock(s, "data/snp_stocks_basic")
        print(s.code)

def snp_stocks_full():
    throttle = 60 * 10
    driver = Stock.get_google_news_driver(headless=True)
    for i in range(0, len(SNP_500_TOP_100)):
        s = Stock(*SNP_500_TOP_100[i], driver=driver)
        s.extract_and_calculate_all(verbose=False)
        save_stock(s, "data/snp_stocks_full")
        print(s.code)
        time.sleep(throttle)

if __name__ == "__main__":
    snp_stocks_full()

# OLD FUNCTIONS

# def get_snp500_stock_arguments():
#     with open("data/snp500_stocks.txt", "r") as file:
#         table = [stock.split("\t") for stock in file.read().split("\n")]
#         headers = table[0]
#         stock_arguments = table[1:]
#         return stock_arguments

# def refresh_saved_stocks_technical_only():
#     stock_arguments = get_snp500_stock_arguments()
#     for i in range(len(stock_arguments)):
#         s = Stock(*stock_arguments[i])
#         s.extract_and_calculate_technical()
#         save_stock(s, "data/old_saved_stocks_technical_only")
#         print(s)

# def get_average_performance(agent, number, episodes, save_fig=True):
#     returns_over_market = [[] for _ in range(episodes)]
#     returns_over_stock = [[] for _ in range(episodes)]
#     for _ in range(number):
#         a = agent()
#         a.train(episodes=episodes)
#         for e in range(episodes):
#             returns_over_market[e].append(a.return_over_market[e])
#             returns_over_stock[e].append(a.return_over_stock[e])
#     average_return_over_market = [sum(vs)/len(vs) for vs in returns_over_market]
#     average_return_over_stock = [sum(vs)/len(vs) for vs in returns_over_stock]
#     # Plot perfomance
#     if save_fig:
#         plt.plot(average_return_over_market, label="Over Market")
#         plt.plot(average_return_over_stock, label="Over Stock")
#         title = f"{number} Agents over {episodes} episodes"
#         plt.title(title)
#         plt.xlabel("Episodes")
#         plt.ylabel("Relative return")
#         plt.legend()
#         plt.savefig(f"data/figs/{title}")
#         plt.clf()
#     return average_return_over_market, average_return_over_stock

# def test_accuracy(environments, date_pairs):
#     def all_agents_trained(environment):
#         agents_dict = dict()
#         # A2C
#         agent = DRLAgent(env = env_train)
#         A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
#         model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
#         agents_dict['a2c'] = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=50000)
#         # PPO
#         agent = DRLAgent(env = env_train)
#         PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
#         agents_dict['ppo'] = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
#         # DDPG
#         agent = DRLAgent(env = env_train)
#         DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
#         model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
#         agents_dict['ddpg'] = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=50000)
#         # SAC
#         agent = DRLAgent(env = env_train)
#         SAC_PARAMS = {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0003,  "learning_starts": 100, "ent_coef": "auto_0.1"}
#         model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
#         agents_dict['sac'] = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=50000)
#         # TD3
#         agent = DRLAgent(env = env_train)
#         TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
#         agents_dict['td3'] = agent.get_model("td3",model_kwargs = TD3_PARAMS)
#         return agents_dict

#     if not os.path.exists("./" + config.DATA_SAVE_DIR):
#         os.makedirs("./" + config.DATA_SAVE_DIR)
#     if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
#         os.makedirs("./" + config.TRAINED_MODEL_DIR)
#     if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
#         os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
#     if not os.path.exists("./" + config.RESULTS_DIR):
#         os.makedirs("./" + config.RESULTS_DIR)
#     # GETTING DATAFRAME
#     df = YahooDownloader(start_date = '2008-01-01',
#                         end_date = '2021-07-01',
#                         ticker_list = config.DOW_30_TICKER).fetch_data()
#     fe = FeatureEngineer(
#                         use_technical_indicator=True,
#                         use_turbulence=False,
#                         user_defined_feature = False)
#     df = fe.preprocess_data(df)
#     # add covariance matrix as states
#     df=df.sort_values(['date','tic'],ignore_index=True)
#     df.index = df.date.factorize()[0]
#     cov_list = []
#     return_list = []
#     # look back is one year
#     lookback=252
#     for i in range(lookback,len(df.index.unique())):
#         data_lookback = df.loc[i-lookback:i,:]
#         price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
#         return_lookback = price_lookback.pct_change().dropna()
#         return_list.append(return_lookback)
#         covs = return_lookback.cov().values 
#         cov_list.append(covs)
#     df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
#     df = df.merge(df_cov, on='date')
#     df = df.sort_values(['date','tic']).reset_index(drop=True)
#     # add grade scores to the dataframe
#     # Get a list of dates
#     dates = df.loc[:,"date"].unique()
#     dates.sort()
#     # Create a new DataFrame for sentiment
#     grade_score_df = pd.DataFrame({"date":[], "tic": [], "grade_score":[]})
#     # For each stock add row for each date
#     for code in config.DOW_30_TICKER:
#         print(code)
#         HEADER = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36'}
#         url = f"https://uk.finance.yahoo.com/quote/{code}/analysis"
#         page = requests.get(url, headers=HEADER)
#         soup = BeautifulSoup(page.content,'html.parser')
#         script = soup.find('script', text=re.compile(r'root\.App\.main'))
#         json_text = re.search(r'^\s*root\.App\.main\s*=\s*({.*?})\s*;\s*$', script.string, flags=re.MULTILINE).group(1)
#         data = json.loads(json_text)
#         rankings_scraped = data['context']['dispatcher']['stores']['QuoteSummaryStore']['upgradeDowngradeHistory']['history']
#         # Create dictionary ranking values
#         rankings_dict = dict()
#         for r in rankings_scraped:
#             investment_ranking = (r['firm'], r['action'], r['fromGrade'], r['toGrade'])
#             # if r['toGrade'] in ["Perform"]:
#             #   print(investment_ranking)
#             if r['toGrade'] in RANKING_VALUES:
#                 value = RANKING_VALUES[r['toGrade']]
#                 date = datetime.datetime.fromtimestamp(r['epochGradeDate']).strftime('%Y-%m-%d')
#                 if date in rankings_dict:
#                     rankings_dict[date].append(value)
#                 else:
#                     rankings_dict[date] = [value]
#         # Add sentiment to dataframe
#         previous_score = 0
#         for index, date in enumerate(dates):
#             if date in rankings_dict:
#                 values = rankings_dict[date]
#                 score = sum(values) / len(values)
#             else:
#                 score = GS_DECAY * previous_score
#             grade_score_df = grade_score_df.append({'date': date, 'tic': code ,'grade_score':score}, ignore_index=True)
#             previous_score = score
#     # Merge new DataFrame with df
#     df =df.merge(grade_score_df, on=['date', 'tic'])

#     # TESTING
#     e_count = 0
#     for e in environments:
#         e_count += 1
#         d_count = 0
#         for dates in date_pairs:
#             d_count += 1
#             # CREATE ENVIRONMENT    
#             train = data_split(df, dates[0][0], dates[0][1])
#             stock_dimension = len(train.tic.unique())
#             state_space = stock_dimension
#             env_kwargs = {
#                 "hmax": 100, 
#                 "initial_amount": 1000000, 
#                 "transaction_cost_pct": 0.001, 
#                 "state_space": state_space, 
#                 "stock_dim": stock_dimension, 
#                 "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
#                 # "all_state_indicators": config.TECHNICAL_INDICATORS_LIST + ['grade_score'],
#                 "action_space": stock_dimension, 
#                 "reward_scaling": 1e-4    
#             }
#             e_train_gym = e(df = train, **env_kwargs)
#             env_train, obs = e_train_gym.get_sb_env()

#             # CREATE AND TRAIN AGENTS
#             # agents_dict = all_agents_trained(env_train)
#             # A2C
#             agents_dict = dict()
#             agent = DRLAgent(env = env_train)
#             A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
#             model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
#             agents_dict['a2c'] = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=500000)

#             for a in agents_dict:
#                 # TEST AGENT PERFORMANCE
#                 trade = data_split(df, dates[1][0], dates[1][1])
#                 e_trade_gym = e(df = trade, **env_kwargs)
#                 df_daily_return, df_actions = DRLAgent.DRL_prediction(model=agents_dict[a], environment=e_trade_gym)
#                 DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
#                 perf_func = timeseries.perf_stats 
#                 perf_stats_all = perf_func(returns=DRL_strat, factor_returns=DRL_strat, positions=None, transactions=None, turnover_denom="AGB")
#                 baseline_df = get_baseline(ticker="^DJI", start=dates[1][0], end=dates[1][1])
#                 print("==============Get Baseline Stats===========")
#                 stats = backtest_stats(baseline_df, value_col_name = 'close')
#                 print("==============DRL Strategy Stats===========")
#                 print(perf_stats_all)

#                 # CREATE A RECORD
#                 with open(f"Agent-{a}_Env-{e_count}_dates-{d_count}.txt", "w") as file:
#                     file.write(f"{e}\n{e.__class__.__name__}\n{str(dates)}\n===Baseline Stats===\n{stats}\n===DRL State===\n{perf_stats_all}")

