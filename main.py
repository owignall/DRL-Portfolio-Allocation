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


def main():
    # get_average_performance(None, 0, 100)
    # get_average_performance(V2SingleStockWeightingDQNAgentWithDNN, 2, 20)

    # refresh_saved_stocks_technical_only()
    
    
    from stable_baselines3 import A2C, DQN

    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCarContinuous-v0')

    env = GymSingleStockWeightControlEnvironment()
    # model = A2C('MlpPolicy', env, verbose=1)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(300):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            env.render()
            obs = env.reset()

if __name__ == "__main__":
    main()


