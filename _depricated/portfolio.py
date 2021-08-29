"""
TO DO
    - Decide on trade request format
    - Complete action method to execute given trades
"""

from stock import *
# import math

ANNUAL_TRADING_DAYS = 253

class Portfolio:
    def __init__(self, stock_universe, start_date_index, end_date_index):
        self.__stock_universe = stock_universe
        self.__start_value = 1_000_000_000 # Note that we are trading in Dollars
        self.start_date_index = start_date_index
        self.date_index = start_date_index
        self.end_date_index = end_date_index
        self.cash = self.__start_value
        self.holdings = dict()
        self.__stock_dictionary = dict()
        for stock in self.__stock_universe:
            self.holdings[stock.code] = 0
            self.__stock_dictionary[stock.code] = stock
        self.buy_all_equally()

    def get_value_from_shares(self, stock, shares):
        return shares * stock.data[self.date_index].adj_close
    
    # def current_holding_value_of_stock(self, stock):
    #     return self.get_value_from_shares(stock, holdings[stock.code])
    
    def get_shares_purchasable_from_value(self, stock, value):
        return value // stock.data[self.date_index].adj_close
    
    def get_shares_sellable_to_raise_value(self, stock, value):
        return math.ceil(value / stock.data[self.date_index].adj_close)

    def buy(self, stock, shares):
        """Buys given number of given stock using cash"""
        if shares > self.get_shares_purchasable_from_value(stock, self.cash):
            raise ValueError(f"Don't have enough cash in portfolio for transaction: Can't buy {shares} shares of {stock.code} at cost of {self.get_value_from_shares(stock, shares)} with {self.cash}")
        else:
            cost_of_transaction = self.get_value_from_shares(stock, shares)
            self.cash -= cost_of_transaction
            self.holdings[stock.code] += shares

    def sell(self, stock, shares):
        """Sells given number of given stock using cash"""
        # number_of_shares = value / stock.data[self.date_index].adj_close
        if shares > self.holdings[stock.code]:
            raise ValueError(f"Don't have enough shares of stock to sell: Can't sell {shares} with only {self.holdings[stock.code]} available")
        else:
            cash_from_sale = self.get_value_from_shares(stock, shares)
            self.holdings[stock.code] -= shares
            self.cash += cash_from_sale

    def sell_given_holdings(self, stocks):
        """Sell all stocks held in the given stocks"""
        for s in stocks:
            self.sell(s, self.holdings[s.code])

    def deploy_cash_evenly(self, stocks):
        """Deploy cash holdings evenly amongst all given stocks"""
        purchase_amount = self.cash / len(stocks)
        for s in stocks:
            shares_to_buy = self.get_shares_purchasable_from_value(s, purchase_amount)
            self.buy(s, shares_to_buy)


    def sell_all_holdings(self):
        for s in self.__stock_universe:
            self.sell(s, self.holdings[s.code])
    
    
    
    def buy_all_equally(self):
        """Deploy capital evenly amongst all shares in stock universe"""
        self.sell_all_holdings()
        purchase_amount = self.cash / len(self.__stock_universe)
        for s in self.__stock_universe:
            shares_to_buy = self.get_shares_purchasable_from_value(s, purchase_amount)
            self.buy(s, shares_to_buy)

    def get_portfolio_value(self):
        """Returns current value of portfolio"""
        value = 0
        value += self.cash
        for stock in self.__stock_universe:
            # value += self.holdings[stock.code] * stock.data[self.date_index].adj_close
            value += self.get_value_from_shares(stock, self.holdings[stock.code])
        return value

    def get_current_daily_data(self):
        """Returns current daily data on every stock in the investment universe"""
        all_daily_data = dict()
        for s in self.__stock_universe:
            all_daily_data[s.code] = s.data[self.date_index]
        return all_daily_data
    
    def action(self, buy_orders, sell_orders):
        """Performs any given trades, increments time by one trading day, and then returns new daily data"""
        pre_step_value = self.get_portfolio_value()
        # Execute buy orders (stock, amount)
        for buy in buy_orders:
            self.buy(*buy)
        for sell in sell_orders:
            self.sell(*sell)
        # Execute sell orders
        self.date_index += 1
        terminal = self.date_index >= self.end_date_index
        post_step_value = self.get_portfolio_value()
        change_in_value = post_step_value - pre_step_value
        return (self.get_current_daily_data(), change_in_value, terminal)
    
    def reset(self):
        """Resets the portfolio to its starting state"""
        self.date_index = self.start_date_index
        self.cash = self.__start_value
        self.holdings = dict()
        for stock in self.__stock_universe:
            self.holdings[stock.code] = 0