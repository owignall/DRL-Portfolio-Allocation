"""
This file contains specialised trading evironments for the agents to interact with directly.
"""

from portfolio import *
from storage import *
import random

ANNUAL_TRADING_DAYS = 253
YEAR_RANGES = [(59 + (year_index*ANNUAL_TRADING_DAYS), 59 + ((year_index+1)*ANNUAL_TRADING_DAYS)) for year_index in range(7)]
TRAINING_YEARS = YEAR_RANGES[:5]
TESTING_YEARS = YEAR_RANGES[5:]

STOCKS = retrieve_stocks_from_folder("data/saved_stocks_technical_only")

class SingleStockEnvironment():
    def __init__(self):
        self.stocks = STOCKS
        self.stock = random.choice(self.stocks)
        self.year_indices = random.choice(TRAINING_YEARS)
        self.portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
        self.trade_size = self.portfolio.get_portfolio_value() // 5
    
    def get_available_actions(self):
        return [0, 1, 2]
    
    def get_stock_weight(self):
        return self.portfolio.get_value_from_shares(self.stock, self.portfolio.holdings[self.stock.code]) / self.portfolio.get_portfolio_value()
    
    def get_state(self):
        weight = self.get_stock_weight()
        daily_data = self.portfolio.get_current_daily_data()[self.stock.code]
        return daily_data.macd, daily_data.signal_line, daily_data.normalized_rsi, daily_data.std_devs_out, daily_data.relative_vol, weight

    def make_action(self, action):
        sells = []
        buys = []
        if action == 0:
            if self.trade_size < self.portfolio.cash:
                amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.trade_size)
            else:
                amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.portfolio.cash)
            buys.append((self.stock, amount))
        elif action == 1:
            shares_to_raise_value = self.portfolio.get_shares_sellable_to_raise_value(self.stock, self.trade_size)
            if shares_to_raise_value < self.portfolio.holdings[self.stock.code]:
                amount = shares_to_raise_value
            else:
                amount = self.portfolio.holdings[self.stock.code]
            sells.append((self.stock, amount))
        elif action == 2:
            pass
        else:
            raise ValueError(f"Trading action '{action}' provided to environment is not valid")
        
        current_value = self.portfolio.get_portfolio_value()
        self.values.append(current_value)
        daily_data, change_in_value, terminal = self.portfolio.action(buys, sells)
        # proportional_value_change = change_in_value / current_value

        return self.get_state(), change_in_value, terminal

    def reset(self):
        self.stock = random.choice(self.stocks)
        self.year_indices = random.choice(TRAINING_YEARS)
        self.portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
        self.trade_size = self.portfolio.get_portfolio_value() // 5
        self.values = [self.portfolio.get_portfolio_value()]

        self.stock_portfolio = Portfolio([self.stock], self.year_indices[0], self.year_indices[1])
        self.stock_portfolio.buy_all_equally()
        self.stock_values = [self.stock_portfolio.get_portfolio_value()]
        terminal = False
        while not terminal:
            _, _, terminal = self.stock_portfolio.action([],[])
            self.stock_values.append(self.stock_portfolio.get_portfolio_value())
        

        return self.get_state(), 0, False
    
class SingleStockWeightControlEnvironment():
    def __init__(self):
        self.stocks = STOCKS
        self.reset()
    
    def reset(self):
        self.stock_index = random.randint(0, len(STOCKS))
        self.stock = STOCKS[self.stock_index]
        self.remaining_stocks = STOCKS[:self.stock_index] + STOCKS[self.stock_index + 1:]
        self.year_indices = random.choice(TRAINING_YEARS)
        # Portfolios
        self.portfolio = Portfolio(STOCKS, self.year_indices[0], self.year_indices[1])
        self.values = [self.portfolio.get_portfolio_value()]
        self.stock_portfolio, self.stock_values = self.generate_benchmark_portfolio([self.stock])
        self.market_portfolio, self.market_values = self.generate_benchmark_portfolio(STOCKS)

        self.trade_size = self.portfolio.get_portfolio_value() // 5
        return self.get_state(), 0, False
    
    def generate_benchmark_portfolio(self, stocks):
        portfolio = Portfolio(stocks, self.year_indices[0], self.year_indices[1])
        values = [portfolio.get_portfolio_value()]
        terminal = False
        while not terminal:
            portfolio.buy_all_equally()
            _, _, terminal = portfolio.action([],[])
            values.append(portfolio.get_portfolio_value())
        return portfolio, values

    def get_available_actions(self):
        return [0, 1, 2]
    
    def get_stock_weight(self):
        return self.portfolio.get_value_from_shares(self.stock, self.portfolio.holdings[self.stock.code]) / self.portfolio.get_portfolio_value()
    
    def get_state(self):
        weight = self.get_stock_weight()
        daily_data = self.portfolio.get_current_daily_data()[self.stock.code]
        return daily_data.macd, daily_data.signal_line, daily_data.normalized_rsi, daily_data.std_devs_out, daily_data.relative_vol, weight

    def make_action(self, action):

        # Sell all holdings other than given stock
        self.portfolio.sell_given_holdings(self.remaining_stocks)

        if action == 0:
            if self.trade_size < self.portfolio.cash:
                amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.trade_size)
            else:
                amount = self.portfolio.get_shares_purchasable_from_value(self.stock, self.portfolio.cash)
            self.portfolio.buy(self.stock, amount)
        elif action == 1:
            shares_to_raise_value = self.portfolio.get_shares_sellable_to_raise_value(self.stock, self.trade_size)
            if shares_to_raise_value < self.portfolio.holdings[self.stock.code]:
                amount = shares_to_raise_value
            else:
                amount = self.portfolio.holdings[self.stock.code]
            self.portfolio.sell(self.stock, amount)
        elif action == 2:
            pass
        else:
            raise ValueError(f"Trading action '{action}' provided to environment is not valid")
        
        # Deploy remaining cash evenly amongst others
        self.portfolio.deploy_cash_evenly(self.remaining_stocks)

        # Calculate value for records
        current_value = self.portfolio.get_portfolio_value()
        self.values.append(current_value)

        # Make take action to move foward a time step
        daily_data, change_in_value, terminal = self.portfolio.action([],[])

        return self.get_state(), change_in_value, terminal

    

if __name__ == "__main__":
    env = SingleStockWeightControlEnvironment()