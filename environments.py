"""
This file contains specialised trading evironments for the agents to interact with directly.
"""

from stock import *
import random

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PortfolioAllocationEnvironment(gym.Env):
    
    def __init__(self, stocks, state_attributes, discrete=False):
        self.check_arguments_valid(stocks, state_attributes)
        self.starting_value = 1_000_000
        self.stocks = [s.reset_index() for s in stocks]
        self.state_attributes = state_attributes
        self.final_index = len(stocks[0]) - 1
        self.final_values = []
        self.annualized_returns = []
        self.sharpe_ratios = []
        self.discrete = discrete

        if self.discrete:
            self.action_space = spaces.Discrete(len(self.stocks))
        else:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.stocks),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state_attributes), len(self.stocks)))
    
    def reset(self):
        self.date_index = 0

        self.portfolio_value = self.starting_value 

        self.state = self.get_state()
        self.terminal = False
        self.return_memory = [0]
        self.value_memory = [self.portfolio_value]
        self.date_memory = [self.stocks[0].loc[self.date_index, 'date']]
        self.actions_memory = [np.array([1 for _ in range(len(self.stocks))])]
        self.reward_memory = []
        return self.state

    def step(self, action):
        if self.terminal: 
            raise Exception("Environment already in terminal state")
        
        
        previous_index = self.date_index
        self.date_index += 1

        if self.discrete:
            s = self.stocks[action]
            previous_close = s.loc[previous_index, 'close']
            new_close = s.loc[self.date_index, 'close']
            portfolio_return = (new_close / previous_close) - 1
        else:
            weights = self.softmax_normalization(action)
            previous_closes = np.array([s.loc[previous_index, 'close'] for s in self.stocks])
            new_closes = np.array([s.loc[self.date_index, 'close'] for s in self.stocks])
            portfolio_return = sum(((new_closes / previous_closes) - 1) * weights)

        change_in_value = self.portfolio_value * portfolio_return
        self.portfolio_value *= (1 + portfolio_return)

        # Memory management
        self.return_memory.append(portfolio_return)
        self.value_memory.append(self.portfolio_value)
        self.date_memory.append(self.stocks[0].loc[self.date_index, 'date'])
        self.actions_memory.append(action)

        self.state = self.get_state()
        self.reward = change_in_value
        self.reward_memory.append(self.reward)

        if self.date_index >= self.final_index: 
            self.terminal = True
            # Performance metrics
            self.final_values.append(self.portfolio_value)
            days = self.final_index + 1
            years = days / TRADING_DAYS
            total_return = (self.portfolio_value / self.starting_value) - 1
            self.annualized_return = ((total_return + 1) ** (1 / years)) - 1
            annualized_volatility = np.std(self.return_memory) * math.sqrt(TRADING_DAYS)
            self.sharpe_ratio = (self.annualized_return - BASE_RATE) / annualized_volatility
            self.sharpe_ratios.append(self.sharpe_ratio)
            self.annualized_returns.append(self.annualized_return)

        return self.state, self.reward, self.terminal, {}

    def get_state(self):
        return np.array([[s.loc[self.date_index, a] for s in self.stocks] for a in self.state_attributes])
    
    @staticmethod
    def softmax_normalization(actions):
        # """Taken from other environment"""
        # numerator = np.exp(actions)
        # denominator = np.sum(np.exp(actions))
        # softmax_output = numerator / denominator
        a = actions.astype('float64')
        return np.exp(actions) /np.sum(np.exp(a))
    
    @staticmethod
    def check_arguments_valid(stocks, state_attributes):
        for stock in stocks:
            if len(stock) != len(stocks[0]):
                raise ValueError("Length of stock DataFrames provided do not match")
            for a in state_attributes:
                if a not in stock.columns:
                    raise ValueError(f"State attribute {a} not available in at least one stock.")


if __name__ == "__main__":
    
    stocks = retrieve_stocks_from_folder("data\snp_stocks_basic")
    # stocks = retrieve_stocks_from_folder("data\snp_stocks_full")

    dfs = [s.df.loc[1000:] for s in stocks[:]]
    env = PortfolioAllocationEnvironment(dfs, ['ranking_score', 'ranking_change_score', 'cheats'])



    obs = env.reset()
    while True:
        action = np.array([1 for _ in range(len(env.stocks))])
        obs, reward, done, info = env.step(action)
        if done:
            break
    print("Even Actions", env.portfolio_value)

    # obs = env.reset()
    # print(obs)
    # while True:
    #     action = [1 for _ in range(len(env.stocks))]
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # print("Even Actions", env.portfolio_value)
    # plt.plot(env.value_memory, label="Even Actions")

    # obs = env.reset()
    # while True:
    #     action = [random.random() for _ in range(len(env.stocks))]
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # print("Random Actions", env.portfolio_value)
    # plt.plot(env.value_memory, label="Random Actions")

    # obs = env.reset()
    # while True:
    #     action = [random.uniform(-5,5) for _ in range(len(env.stocks))]
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # print("Very Random Actions", env.portfolio_value)
    # plt.plot(env.value_memory, label="Very Random Actions")

    
    # if len(obs) >= 3:
    #     sigmoid_v = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    #     obs = env.reset()
    #     while True:
    #         # action = sigmoid_v(obs[0])
    #         action = obs[0]
    #         obs, reward, done, info = env.step(action)
    #         if done:
    #             break
    #     print("New Score Actions", env.portfolio_value)
    #     plt.plot(env.value_memory, label="New Score Actions")


    # obs = env.reset()
    # while True:
    #     # action = sigmoid_v(obs[1])
    #     action = obs[1] * 3
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # print("Change Score Actions", env.portfolio_value)
    # plt.plot(env.value_memory, label="Change Score Actions")

    # obs = env.reset()
    # while True:
    #     # action = sigmoid_v(obs[1])
    #     action = obs[2] * 3
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # print("Cheat Actions", env.portfolio_value)
    # plt.plot(env.value_memory, label="Cheat Actions")


    # title = f"Comparison {len(env.stocks)}"
    # plt.title(title)
    # plt.xlabel("Steps")
    # plt.ylabel("Return")
    # plt.legend()
    # plt.savefig(f"{title}")
    # plt.clf()


# OLD STUFF

# class SingleStockEnvironment():
#     def __init__(self):
#         self.stocks = STOCKS
#         self.stock = random.choice(self.stocks)
#         self.year_indices = random.choice(TRAINING_YEARS)
#         self.portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
#         self.trade_size = self.portfolio.get_portfolio_value() // 5
    
#     def get_available_actions(self):
#         return [0, 1, 2]
    
#     def get_stock_weight(self):
#         return self.portfolio.get_value_from_shares(self.stock, self.portfolio.holdings[self.stock.code]) / self.portfolio.get_portfolio_value()
    
#     def get_state(self):
#         weight = self.get_stock_weight()
#         daily_data = self.portfolio.get_current_daily_data()[self.stock.code]
#         return daily_data.macd, daily_data.signal_line, daily_data.normalized_rsi, daily_data.std_devs_out, daily_data.relative_vol, weight

#     def make_action(self, action):
#         sells = []
#         buys = []
#         if action == 0:
#             if self.trade_size < self.portfolio.cash:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.trade_size)
#             else:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.portfolio.cash)
#             buys.append((self.stock, amount))
#         elif action == 1:
#             shares_to_raise_value = self.portfolio.get_shares_sellable_to_raise_value(self.stock, self.trade_size)
#             if shares_to_raise_value < self.portfolio.holdings[self.stock.code]:
#                 amount = shares_to_raise_value
#             else:
#                 amount = self.portfolio.holdings[self.stock.code]
#             sells.append((self.stock, amount))
#         elif action == 2:
#             pass
#         else:
#             raise ValueError(f"Trading action '{action}' provided to environment is not valid")
        
#         current_value = self.portfolio.get_portfolio_value()
#         self.values.append(current_value)
#         daily_data, change_in_value, terminal = self.portfolio.action(buys, sells)
#         # proportional_value_change = change_in_value / current_value

#         return self.get_state(), change_in_value, terminal

#     def reset(self):
#         self.stock = random.choice(self.stocks)
#         self.year_indices = random.choice(TRAINING_YEARS)
#         self.portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
#         self.trade_size = self.portfolio.get_portfolio_value() // 5
#         self.values = [self.portfolio.get_portfolio_value()]

#         self.stock_portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
#         self.stock_portfolio.buy_all_equally()
#         self.stock_values = [self.stock_portfolio.get_portfolio_value()]
#         terminal = False
#         while not terminal:
#             _, _, terminal = self.stock_portfolio.action([],[])
#             self.stock_values.append(self.stock_portfolio.get_portfolio_value())
        

#         return self.get_state(), 0, False
    
# class SingleStockWeightControlEnvironment():
#     def __init__(self):
#         self.stocks = STOCKS
#         self.reset()
    
#     def reset(self):
#         self.stock_index = random.randint(0, len(STOCKS) - 1)
#         self.stock = STOCKS[self.stock_index]
#         self.remaining_stocks = STOCKS[:self.stock_index] + STOCKS[self.stock_index + 1:]
#         self.year_indices = random.choice(TRAINING_YEARS)

#         # TEMP TO MAKE EASIER
#         self.stock = STOCKS[10]
#         self.year_indices = TRAINING_YEARS[0]

#         # Portfolios
#         self.portfolio = Portfolio(STOCKS, self.year_indices[0], self.year_indices[1])
#         self.values = [self.portfolio.get_portfolio_value()]
#         self.stock_portfolio, self.stock_values = self.generate_benchmark_portfolio([self.stock])
#         self.market_portfolio, self.market_values = self.generate_benchmark_portfolio(STOCKS)

#         self.trade_size = self.portfolio.get_portfolio_value()
#         return self.get_state(), 0, False
    
#     def generate_benchmark_portfolio(self, stocks):
#         portfolio = Portfolio(stocks, self.year_indices[0], self.year_indices[1])
#         values = [portfolio.get_portfolio_value()]
#         terminal = False
#         while not terminal:
#             portfolio.buy_all_equally()
#             _, _, terminal = portfolio.action([],[])
#             values.append(portfolio.get_portfolio_value())
#         return portfolio, values

#     def get_available_actions(self):
#         return [0, 1, 2]
    
#     def get_stock_weight(self):
#         return self.portfolio.get_value_from_shares(self.stock, self.portfolio.holdings[self.stock.code]) / self.portfolio.get_portfolio_value()
    
#     def get_state(self):
#         weight = self.get_stock_weight()
#         daily_data = self.portfolio.get_current_daily_data()[self.stock.code]
#         return daily_data.macd, daily_data.signal_line, daily_data.normalized_rsi, daily_data.std_devs_out, daily_data.relative_vol, weight

#     def make_action(self, action):

#         # Sell all holdings other than given stock
#         self.portfolio.sell_given_holdings(self.remaining_stocks)

#         if action == 0:
#             if self.trade_size < self.portfolio.cash:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.trade_size)
#             else:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.portfolio.cash)
#             self.portfolio.buy(self.stock, amount)
#         elif action == 1:
#             shares_to_raise_value = self.portfolio.get_shares_sellable_to_raise_value(self.stock, self.trade_size)
#             if shares_to_raise_value < self.portfolio.holdings[self.stock.code]:
#                 amount = shares_to_raise_value
#             else:
#                 amount = self.portfolio.holdings[self.stock.code]
#             self.portfolio.sell(self.stock, amount)
#         elif action == 2:
#             pass
#         else:
#             raise ValueError(f"Trading action '{action}' provided to environment is not valid")
        
#         # Deploy remaining cash evenly amongst others
#         self.portfolio.deploy_cash_evenly(self.remaining_stocks)

#         # Calculate value for records
#         current_value = self.portfolio.get_portfolio_value()
#         self.values.append(current_value)

#         # Make take action to move foward a time step
#         daily_data, change_in_value, terminal = self.portfolio.action([],[])

#         return self.get_state(), change_in_value, terminal

# # Gym complient environments

# class GymSingleStockWeightControlEnvironment(gym.Env):
#     def __init__(self):
#         self.stocks = STOCKS
#         self.fixed_stock_index = random.randint(0, len(STOCKS) - 1)
#         # Gym attributes
#         observation_high = np.array([np.finfo(np.float32).max] * 1) # TEMP CHANGE
#         self.action_space = gym.spaces.Discrete(3)
#         self.observation_space = gym.spaces.Box(low=-observation_high, high=observation_high)
        
#     def reset(self):
#         self.stock_index = random.randint(0, len(STOCKS) - 1)
#         self.stock = STOCKS[self.stock_index]
#         self.remaining_stocks = STOCKS[:self.stock_index] + STOCKS[self.stock_index + 1:]
#         self.year_indices = random.choice(TRAINING_YEARS)

#         # TEMP TO MAKE EASIER
#         self.stock = STOCKS[self.fixed_stock_index]
#         self.year_indices = TRAINING_YEARS[0]

#         # Portfolios
#         self.portfolio = Portfolio(STOCKS, self.year_indices[0], self.year_indices[1])
#         self.values = [self.portfolio.get_portfolio_value()]
#         self.stock_portfolio, self.stock_values = self.generate_benchmark_portfolio([self.stock])
#         self.market_portfolio, self.market_values = self.generate_benchmark_portfolio(STOCKS)

#         self.trade_size = self.portfolio.get_portfolio_value()

#         self.time_step = 0
#         return np.array(self.get_state())
    
#     def generate_benchmark_portfolio(self, stocks):
#         portfolio = Portfolio(stocks, self.year_indices[0], self.year_indices[1])
#         values = [portfolio.get_portfolio_value()]
#         terminal = False
#         while not terminal:
#             portfolio.buy_all_equally()
#             _, _, terminal = portfolio.action([],[])
#             values.append(portfolio.get_portfolio_value())
#         return portfolio, values

#     def get_available_actions(self):
#         return [0, 1, 2]
    
#     def get_stock_weight(self):
#         return self.portfolio.get_value_from_shares(self.stock, self.portfolio.holdings[self.stock.code]) / self.portfolio.get_portfolio_value()
    
#     def get_state(self):
#         weight = self.get_stock_weight()
#         daily_data = self.portfolio.get_current_daily_data()[self.stock.code]
#         # TEMP
#         # return daily_data.macd, daily_data.signal_line, daily_data.normalized_rsi, daily_data.std_devs_out, daily_data.relative_vol, weight, self.time_step/253
#         return (self.time_step/253,)

#     def step(self, action):
#         self.time_step += 1
#         # Sell all holdings other than given stock
#         self.portfolio.sell_given_holdings(self.remaining_stocks)

#         if action == 0:
#             if self.trade_size < self.portfolio.cash:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.trade_size)
#             else:
#                 amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.portfolio.cash)
#             self.portfolio.buy(self.stock, amount)
#         elif action == 1:
#             shares_to_raise_value = self.portfolio.get_shares_sellable_to_raise_value(self.stock, self.trade_size)
#             if shares_to_raise_value < self.portfolio.holdings[self.stock.code]:
#                 amount = shares_to_raise_value
#             else:
#                 amount = self.portfolio.holdings[self.stock.code]
#             self.portfolio.sell(self.stock, amount)
#         elif action == 2:
#             pass
#         else:
#             raise ValueError(f"Trading action '{action}' provided to environment is not valid")
        
#         # Deploy remaining cash evenly amongst others
#         self.portfolio.deploy_cash_evenly(self.remaining_stocks)

#         # Calculate value for records
#         current_value = self.portfolio.get_portfolio_value()
#         self.values.append(current_value)

#         # Make take action to move foward a time step
#         daily_data, change_in_value, terminal = self.portfolio.action([],[])

#         return np.array(self.get_state()), change_in_value, terminal, {}
    
#     def render(self):
#         plt.plot(self.values, label="Portfolio")
#         plt.plot(self.stock_values, label="Stock")
#         plt.plot(self.market_values, label="Market")
#         title = f"Return over episode"
#         plt.title(title)
#         plt.xlabel("Steps")
#         plt.ylabel("Return")
#         plt.legend()
#         # plt.savefig(f"data/figs/{title}")
#         plt.show()
#         plt.clf()

# # Adapted versions of provided environments

# class TIOnlyStockPortfolioEnv(gym.Env):
#     """
#     A portfolio stock trading environment for OpenAI gym
#     """

#     metadata = {"render.modes": ["human"]}

#     def __init__(
#         self,
#         df,
#         stock_dim,
#         hmax,
#         initial_amount,
#         transaction_cost_pct,
#         reward_scaling,
#         state_space,
#         action_space,
#         tech_indicator_list,
#         turbulence_threshold=None,
#         lookback=252,
#         day=0,
#     ):
#         # super(StockEnv, self).__init__()
#         # money = 10 , scope = 1
#         self.day = day
#         self.lookback = lookback
#         self.df = df
#         self.stock_dim = stock_dim
#         self.hmax = hmax
#         self.initial_amount = initial_amount
#         self.transaction_cost_pct = transaction_cost_pct
#         self.reward_scaling = reward_scaling
#         self.state_space = state_space
#         self.action_space = action_space
#         self.tech_indicator_list = tech_indicator_list

#         # action_space normalization and shape is self.stock_dim
#         self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
#         # Shape = (4, 30)
#         # covariance matrix + technical indicators
#         self.observation_space = spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(len(self.tech_indicator_list), self.state_space),
#         )

#         # load data from a pandas dataframe
#         self.data = self.df.loc[self.day, :]
#         # self.covs = self.data["cov_list"].values[0]
#         self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])


#         self.terminal = False
#         self.turbulence_threshold = turbulence_threshold
#         # initalize state: inital portfolio return + individual stock return + individual weights
#         self.portfolio_value = self.initial_amount

#         # memorize portfolio value each step
#         self.asset_memory = [self.initial_amount]
#         # memorize portfolio return each step
#         self.portfolio_return_memory = [0]
#         self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
#         self.date_memory = [self.data.date.unique()[0]]

#     def step(self, actions):
#         # print(self.day)
#         self.terminal = self.day >= len(self.df.index.unique()) - 1
#         # print(actions)

#         if self.terminal:
#             df = pd.DataFrame(self.portfolio_return_memory)
#             df.columns = ["daily_return"]
#             # plt.plot(df.daily_return.cumsum(), "r")
#             # plt.savefig("results/cumulative_reward.png")
#             # plt.close()

#             # plt.plot(self.portfolio_return_memory, "r")
#             # plt.savefig("results/rewards.png")
#             # plt.close()

#             print("=================================")
#             print("begin_total_asset:{}".format(self.asset_memory[0]))
#             print("end_total_asset:{}".format(self.portfolio_value))

#             df_daily_return = pd.DataFrame(self.portfolio_return_memory)
#             df_daily_return.columns = ["daily_return"]
#             if df_daily_return["daily_return"].std() != 0:
#                 sharpe = (
#                     (252 ** 0.5)
#                     * df_daily_return["daily_return"].mean()
#                     / df_daily_return["daily_return"].std()
#                 )
#                 print("Sharpe: ", sharpe)
#             print("=================================")

#             return self.state, self.reward, self.terminal, {}

#         else:
#             # print("Model actions: ",actions)
#             # actions are the portfolio weight
#             # normalize to sum of 1
#             # if (np.array(actions) - np.array(actions).min()).sum() != 0:
#             #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
#             # else:
#             #  norm_actions = actions
#             weights = self.softmax_normalization(actions)
#             # print("Normalized actions: ", weights)
#             self.actions_memory.append(weights)
#             last_day_memory = self.data

#             # load next state
#             self.day += 1
#             self.data = self.df.loc[self.day, :]
#             # self.covs = self.data["cov_list"].values[0]
#             self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
#             # print(self.state)
#             # calcualte portfolio return
#             # individual stocks' return * weight
#             portfolio_return = sum(
#                 ((self.data.close.values / last_day_memory.close.values) - 1) * weights
#             )
#             # update portfolio value
#             new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
#             self.portfolio_value = new_portfolio_value

#             # save into memory
#             self.portfolio_return_memory.append(portfolio_return)
#             self.date_memory.append(self.data.date.unique()[0])
#             self.asset_memory.append(new_portfolio_value)

#             # the reward is the new portfolio value or end portfolo value
#             self.reward = new_portfolio_value
#             # print("Step reward: ", self.reward)
#             # self.reward = self.reward*self.reward_scaling

#         return self.state, self.reward, self.terminal, {}

#     def reset(self):
#         self.asset_memory = [self.initial_amount]
#         self.day = 0
#         self.data = self.df.loc[self.day, :]
#         # load states
#         # self.covs = self.data["cov_list"].values[0]
#         self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
#         self.portfolio_value = self.initial_amount
#         # self.cost = 0
#         # self.trades = 0
#         self.terminal = False
#         self.portfolio_return_memory = [0]
#         self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
#         self.date_memory = [self.data.date.unique()[0]]
#         return self.state

#     def render(self, mode="human"):
#         return self.state

#     def softmax_normalization(self, actions):
#         numerator = np.exp(actions)
#         denominator = np.sum(np.exp(actions))
#         softmax_output = numerator / denominator
#         return softmax_output

#     def save_asset_memory(self):
#         date_list = self.date_memory
#         portfolio_return = self.portfolio_return_memory
#         # print(len(date_list))
#         # print(len(asset_list))
#         df_account_value = pd.DataFrame(
#             {"date": date_list, "daily_return": portfolio_return}
#         )
#         return df_account_value

#     def save_action_memory(self):
#         # date and close price length must match actions length
#         date_list = self.date_memory
#         df_date = pd.DataFrame(date_list)
#         df_date.columns = ["date"]

#         action_list = self.actions_memory
#         df_actions = pd.DataFrame(action_list)
#         df_actions.columns = self.data.tic.values
#         df_actions.index = df_date.date
#         # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
#         return df_actions

#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def get_sb_env(self):
#         e = DummyVecEnv([lambda: self])
#         obs = e.reset()
#         return e, obs


# class GSAndTIStockPortfolioEnv(gym.Env):
#     """
#     A portfolio stock trading environment for OpenAI gym
#     """
#     metadata = {"render.modes": ["human"]}

#     def __init__(
#         self,
#         df,
#         stock_dim,
#         hmax,
#         initial_amount,
#         transaction_cost_pct,
#         reward_scaling,
#         state_space,
#         action_space,
#         tech_indicator_list,
#         all_state_indicators,
#         turbulence_threshold=None,
#         lookback=252,
#         day=0,
#         ):
#         # super(StockEnv, self).__init__()
#         # money = 10 , scope = 1
#         self.day = day
#         self.lookback = lookback
#         self.df = df
#         self.stock_dim = stock_dim
#         self.hmax = hmax
#         self.initial_amount = initial_amount
#         self.transaction_cost_pct = transaction_cost_pct
#         self.reward_scaling = reward_scaling
#         self.state_space = state_space
#         self.action_space = action_space
#         self.tech_indicator_list = tech_indicator_list
#         self.all_state_indicators = all_state_indicators

#         # action_space normalization and shape is self.stock_dim
#         self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
#         # Shape = (4, 30)
#         # covariance matrix + technical indicators
#         self.observation_space = spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(len(self.all_state_indicators), self.state_space),
#         )

#         # load data from a pandas dataframe
#         self.data = self.df.loc[self.day, :]
#         # self.covs = self.data["cov_list"].values[0]
#         self.state = np.array([self.data[tech].values.tolist() for tech in self.all_state_indicators])


#         self.terminal = False
#         self.turbulence_threshold = turbulence_threshold
#         # initalize state: inital portfolio return + individual stock return + individual weights
#         self.portfolio_value = self.initial_amount

#         # memorize portfolio value each step
#         self.asset_memory = [self.initial_amount]
#         # memorize portfolio return each step
#         self.portfolio_return_memory = [0]
#         self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
#         self.date_memory = [self.data.date.unique()[0]]

#     def step(self, actions):
#         # print(self.day)
#         self.terminal = self.day >= len(self.df.index.unique()) - 1
#         # print(actions)

#         if self.terminal:
#             df = pd.DataFrame(self.portfolio_return_memory)
#             df.columns = ["daily_return"]
#             # plt.plot(df.daily_return.cumsum(), "r")
#             # plt.savefig("results/cumulative_reward.png")
#             # plt.close()

#             # plt.plot(self.portfolio_return_memory, "r")
#             # plt.savefig("results/rewards.png")
#             # plt.close()

#             print("=================================")
#             print("begin_total_asset:{}".format(self.asset_memory[0]))
#             print("end_total_asset:{}".format(self.portfolio_value))

#             df_daily_return = pd.DataFrame(self.portfolio_return_memory)
#             df_daily_return.columns = ["daily_return"]
#             if df_daily_return["daily_return"].std() != 0:
#                 sharpe = (
#                     (252 ** 0.5)
#                     * df_daily_return["daily_return"].mean()
#                     / df_daily_return["daily_return"].std()
#                 )
#                 print("Sharpe: ", sharpe)
#             print("=================================")

#             return self.state, self.reward, self.terminal, {}

#         else:
#             # print("Model actions: ",actions)
#             # actions are the portfolio weight
#             # normalize to sum of 1
#             # if (np.array(actions) - np.array(actions).min()).sum() != 0:
#             #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
#             # else:
#             #  norm_actions = actions
#             weights = self.softmax_normalization(actions)
#             # print("Normalized actions: ", weights)
#             self.actions_memory.append(weights)
#             last_day_memory = self.data

#             # load next state
#             self.day += 1
#             self.data = self.df.loc[self.day, :]
#             # self.covs = self.data["cov_list"].values[0]
#             self.state = np.array([self.data[tech].values.tolist() for tech in self.all_state_indicators])
#             # print(self.state)
#             # calcualte portfolio return
#             # individual stocks' return * weight
#             portfolio_return = sum(
#                 ((self.data.close.values / last_day_memory.close.values) - 1) * weights
#             )
#             # update portfolio value
#             new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
#             self.portfolio_value = new_portfolio_value

#             # save into memory
#             self.portfolio_return_memory.append(portfolio_return)
#             self.date_memory.append(self.data.date.unique()[0])
#             self.asset_memory.append(new_portfolio_value)

#             # the reward is the new portfolio value or end portfolo value
#             self.reward = new_portfolio_value
#             # print("Step reward: ", self.reward)
#             # self.reward = self.reward*self.reward_scaling

#         return self.state, self.reward, self.terminal, {}

#     def reset(self):
#         self.asset_memory = [self.initial_amount]
#         self.day = 0
#         self.data = self.df.loc[self.day, :]
#         # load states
#         # self.covs = self.data["cov_list"].values[0]
        
#         self.state = np.array([self.data[tech].values.tolist() for tech in self.all_state_indicators])
#         self.portfolio_value = self.initial_amount
#         # self.cost = 0
#         # self.trades = 0
#         self.terminal = False
#         self.portfolio_return_memory = [0]
#         self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
#         self.date_memory = [self.data.date.unique()[0]]
#         return self.state

#     def render(self, mode="human"):
#         return self.state

#     def softmax_normalization(self, actions):
#         numerator = np.exp(actions)
#         denominator = np.sum(np.exp(actions))
#         softmax_output = numerator / denominator
#         return softmax_output

#     def save_asset_memory(self):
#         date_list = self.date_memory
#         portfolio_return = self.portfolio_return_memory
#         # print(len(date_list))
#         # print(len(asset_list))
#         df_account_value = pd.DataFrame(
#             {"date": date_list, "daily_return": portfolio_return}
#         )
#         return df_account_value

#     def save_action_memory(self):
#         # date and close price length must match actions length
#         date_list = self.date_memory
#         df_date = pd.DataFrame(date_list)
#         df_date.columns = ["date"]

#         action_list = self.actions_memory
#         df_actions = pd.DataFrame(action_list)
#         df_actions.columns = self.data.tic.values
#         df_actions.index = df_date.date
#         # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
#         return df_actions

#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def get_sb_env(self):
#         e = DummyVecEnv([lambda: self])
#         obs = e.reset()
#         return e, obs

# # Custom environments that comply with FinRL

# class V1(gym.Env): # Note this may be different in the notebook 
#     def __init__(self, training=True):
#         if training:
#             self.years = TRAINING_YEARS
#         else:
#             self.years = TESTING_YEARS
#         self.start_index = self.years[0][0]
#         self.end_index = self.years[-1][-1]

#         self.stocks = STOCKS[0:30]
#         self.state_attributes = ["macd", "signal_line", "normalized_rsi", "std_devs_out", "relative_vol"]

#         self.action_space = spaces.Box(low=0, high=1, shape=(len(self.stocks),))
#         self.observation_space = spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(len(self.state_attributes), len(self.stocks)),
#         )

    
#     def step(self, actions):
#         weights = self.softmax_normalization(actions)
#         previous_index = self.date_index

#         self.date_index += 1

#         previous_closes = np.array([stock.data[previous_index].close for stock in self.stocks])
#         new_closes = np.array([stock.data[self.date_index].close for stock in self.stocks])
#         portfolio_return = sum(((new_closes / previous_closes) - 1) * weights)
#         self.portfolio_value *= (1 + portfolio_return)

#         self.return_memory.append(portfolio_return)
#         self.value_memory.append(self.portfolio_value)
#         self.date_memory.append(self.stocks[0].data[self.date_index].date)
#         self.actions_memory.append(actions)

#         self.state = self.get_state()
#         self.reward = self.portfolio_value
#         if self.date_index > self.end_index: self.terminal = True

#         return self.state, self.reward, self.terminal, {}

#     def reset(self):
#         self.date_index = self.start_index
#         self.portfolio_value = 1_000_000

#         self.state = self.get_state()
#         self.terminal = False
#         self.return_memory = [0]
#         self.value_memory = [self.portfolio_value]
#         self.date_memory = [self.stocks[0].data[self.date_index].date]
#         self.actions_memory = [np.array([1 for _ in range(len(self.stocks))])]

#         return self.state
    
#     def get_state(self):
#         return np.array([[getattr(stock.data[self.date_index], a) for stock in self.stocks] for a in self.state_attributes])

#     def softmax_normalization(self, actions):
#         """Taken from other environment"""
#         numerator = np.exp(actions)
#         denominator = np.sum(np.exp(actions))
#         softmax_output = numerator / denominator
#         return softmax_output
    
#     def save_asset_memory(self):
#         date_list = self.date_memory
#         portfolio_return = self.return_memory
#         df_account_value = pd.DataFrame(
#             {"date": date_list, "daily_return": portfolio_return}
#         )
#         return df_account_value

#     def save_action_memory(self):
#         # date and close price length must match actions length
#         date_list = self.date_memory
#         df_date = pd.DataFrame(date_list)
#         df_date.columns = ["date"]

#         action_list = self.actions_memory
#         df_actions = pd.DataFrame(action_list)
#         df_actions.columns = [s.code for s in self.stocks]# self.data.tic.values
#         df_actions.index = df_date.date
#         # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
#         return df_actions

#     def get_sb_env(self):
#         e = DummyVecEnv([lambda: self])
#         obs = e.reset()
#         return e, obs

