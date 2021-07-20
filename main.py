"""
Plan for this file.
    1. Create portfolio object from universe of presaved stocks
    2. Create agent objects
    3. Train agents.
    4. Test agent performance
"""

from agents import *
from storage import *
import matplotlib.pyplot as plt

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
    print("ROM", len(returns_over_market))
    returns_over_stock = [[] for _ in range(episodes)]
    for _ in range(number):
        a = agent()
        a.train(episodes=episodes)
        for e in range(episodes):
            print("e", e)
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


def main():
    # get_average_performance(None, 0, 100)
    get_average_performance(V2SingleStockWeightingDQNAgentWithDNN, 2, 20)

    # refresh_saved_stocks_technical_only()
    # stocks = retrieve_stocks_from_folder("data/saved_stocks_technical_only")



    # agent = SingleStockDQNAgentWithDNN()
    # agent.train()





    # p1 = Portfolio(stocks)
    # agent1 = BenchmarkAgent(p1)
    # agent1.run_episode()
    # # print(agent1.values)

    # p2 = Portfolio(stocks)
    # agent2 = SplitAndHoldAgent(p2)
    # agent2.run_episode()
    # # print(agent2.values)

    # # plt.figure(figsize =(12,8))
    # plt.plot(agent1.values)
    # plt.plot(agent2.values)
    # # plt.title(f"{number_of_agents} Sarsa Agents")
    # # plt.xlabel("Episodes")
    # # plt.ylabel("Average Undiscounted Return")
    # plt.show() 

if __name__ == "__main__":
    main()


