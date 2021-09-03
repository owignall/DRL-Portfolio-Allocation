"""
This is the environment class that can be used by deep reinforcement learning agents. It
follows the Gym framework and can be instantiated with either a discrete or continuous action
space. It also tracks the performance of all the episodes run on it.
"""

from stock import *

import random
import numpy as np
import pandas as pd
import gym
from gym import spaces


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
        self.reset()
        
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