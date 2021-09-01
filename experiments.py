"""

"""

from environments import *

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
from datetime import datetime as dt

from stable_baselines3 import A2C, DQN


def e1_parameter_search():
    path = f"data/results/raw_parameter_search/"  
    experiment_values = [
        (0.001, 0.99), (0.001, 0.9), (0.001, 0.7), (0.001, 0), (0.0001, 0), 
        (0.0005, 0), (0.005, 0), (0.01, 0), (0.0003, 0), (0.0004, 0), (0.0006, 0), (0.0007, 0)
    ]
    repeats = 10
    total_training_steps = 100_000
    attributes = ['cheats']
    stocks = retrieve_stocks_from_folder("data/snp_stocks_basic")
    train_dfs = [s.df.loc[:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Testing Baseline
    testing_results = pd.DataFrame({"Episode": [1]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=0.001, gamma=0)
        test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value]
    testing_results.to_excel(path + f"untrained_testing.xlsx")

    for alpha, gamma in experiment_values:
        print(f"Alpha = {alpha}, Gamma = {gamma}")
        training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
        testing_results = pd.DataFrame({"Episode": [1]})
        for i in range(repeats):
            print(f"Repeat {i + 1}")
            train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
            train_env.reset()
            model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
            model.learn(total_timesteps=total_training_steps)

            training_results[i + 1] = train_env.final_values

            test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
            obs = test_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break

            testing_results[i + 1] = [test_env.portfolio_value]
                  
        training_results.to_excel(path + f"{alpha}_{gamma}_training.xlsx")
        testing_results.to_excel(path + f"{alpha}_{gamma}_testing.xlsx")

def e2_technical_indicators():
    path = "data/results/raw_technical_indicators/"
    test_attributes = [
        ['macd'],
        ['signal_line'],
        ['normalized_rsi'],
        ['std_devs_out'],
        ['relative_vol'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol']
    ]

    repeats = 10
    total_training_steps = 150_000
    alpha = 0.0005
    gamma = 0
    stocks = retrieve_stocks_from_folder("data/snp_stocks_basic")

    train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Test Baseline
    testing_results = pd.DataFrame({"Episode": [1]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, ['macd'])
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
        test_env = PortfolioAllocationEnvironment(test_dfs, ['macd'])
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value]
    testing_results.to_excel(path + f"untrained_testing.xlsx")

    # Agents
    for attributes in test_attributes:
        print(f"Attributes = {attributes}")
        training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
        testing_results = pd.DataFrame({"Episode": [1]})
        for i in range(repeats):
            print(f"Repeat {i + 1}")
            train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
            train_env.reset()
            model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
            model.learn(total_timesteps=total_training_steps)

            training_results[i + 1] = train_env.final_values

            test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
            obs = test_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break

            testing_results[i + 1] = [test_env.portfolio_value]
        
        seperator = "-"        
        training_results.to_excel(path + f"{seperator.join(attributes)}_training.xlsx")
        testing_results.to_excel(path + f"{seperator.join(attributes)}_testing.xlsx")

def e3_sentiment_features():
    path = "data/results/raw_sentiment_features/"
    test_attributes = [
        ['hf_google_articles_score'],
        ['tb_google_articles_score'], 
        ['vader_google_articles_score'],
        ['ranking_change_score'], 
        ['ranking_score'],
        ['hf_google_articles_score', 'tb_google_articles_score', 'vader_google_articles_score', 'ranking_change_score', 'ranking_score']
    ]
    repeats = 10
    total_training_steps = 150_000
    alpha = 0.0005
    gamma = 0
    stocks = retrieve_stocks_from_folder("data/snp_50_stocks_full")

    train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Test Baseline
    testing_results = pd.DataFrame({"Episode": [1]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, ['macd'])
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
        test_env = PortfolioAllocationEnvironment(test_dfs, ['macd'])
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value]
    testing_results.to_excel(path + f"untrained_testing.xlsx")
        
    # Agents
    for attributes in test_attributes:
        print(f"Attributes = {attributes}")
        training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
        testing_results = pd.DataFrame({"Episode": [1]})
        for i in range(repeats):
            print(f"Repeat {i + 1}")
            train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
            train_env.reset()
            model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
            model.learn(total_timesteps=total_training_steps)

            training_results[i + 1] = train_env.final_values

            test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
            obs = test_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break

            testing_results[i + 1] = [test_env.portfolio_value]
        
        seperator = "-"        
        training_results.to_excel(path + f"{seperator.join(attributes)}_training.xlsx")
        testing_results.to_excel(path + f"{seperator.join(attributes)}_testing.xlsx")

def e4_combined_features():
    path = "data/results/raw_combined_features/"
    test_attributes = [
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol'],
        ['hf_google_articles_score', 'tb_google_articles_score', 'vader_google_articles_score', 'ranking_change_score', 'ranking_score'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol', 'hf_google_articles_score', 
            'tb_google_articles_score', 'vader_google_articles_score', 'ranking_change_score', 'ranking_score']
    ]
    repeats = 10
    total_training_steps = 300_000
    alpha = 0.0005
    gamma = 0
    stocks = retrieve_stocks_from_folder("data/snp_50_stocks_full")

    train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Test Baseline
    testing_results = pd.DataFrame({"Episode": [1]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, ['macd'])
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
        test_env = PortfolioAllocationEnvironment(test_dfs, ['macd'])
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value]
    testing_results.to_excel(path + f"untrained_testing.xlsx")
        
    # Agents
    for attributes in test_attributes:
        print(f"Attributes = {attributes}")
        training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
        testing_results = pd.DataFrame({"Episode": [1]})
        for i in range(repeats):
            print(f"Repeat {i + 1}")
            train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
            train_env.reset()
            model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
            model.learn(total_timesteps=total_training_steps)

            training_results[i + 1] = train_env.final_values

            test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
            obs = test_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break

            testing_results[i + 1] = [test_env.portfolio_value]
        
        seperator = "-"        
        training_results.to_excel(path + f"{seperator.join(attributes)}_training.xlsx")
        testing_results.to_excel(path + f"{seperator.join(attributes)}_testing.xlsx")

def e4_2_combined_features_refined():
    path = "data/results/raw_combined_features_refined_3/"

    test_attributes = [
        ['normalized_rsi', 'std_devs_out', 'relative_vol'],
        ['hf_google_articles_score', 'vader_google_articles_score', 'ranking_score'],
        ['normalized_rsi', 'std_devs_out', 'relative_vol',
            'hf_google_articles_score', 'vader_google_articles_score', 'ranking_score']
    ]
    repeats = 5
    total_training_steps = 500_000
    alpha = 0.0005
    gamma = 0
    stocks = retrieve_stocks_from_folder("data/snp_50_stocks_full")

    train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Test Baseline
    testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, ['macd'])
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
        test_env = PortfolioAllocationEnvironment(test_dfs, ['macd'])
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
    testing_results.to_excel(path + f"untrained_testing.xlsx")
        
    # Agents
    for attributes in test_attributes:
        print(f"Attributes = {attributes}")
        training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
        testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
        for i in range(repeats):
            print(f"Repeat {i + 1}")
            train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
            train_env.reset()
            model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=alpha, gamma=gamma)
            model.learn(total_timesteps=total_training_steps)

            training_results[i + 1] = train_env.final_values

            test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
            obs = test_env.reset()
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break

            testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
        
        seperator = "-"        
        training_results.to_excel(path + f"{seperator.join(attributes)}_training.xlsx")
        testing_results.to_excel(path + f"{seperator.join(attributes)}_testing.xlsx")

def e5_model_comparison():
    path = "data/results/raw_model_comparison/"

    attributes = ['normalized_rsi', 'std_devs_out', 'relative_vol', 
        'hf_google_articles_score', 'vader_google_articles_score', 'ranking_score']
    repeats = 20
    total_training_steps = 300_000
    stocks = retrieve_stocks_from_folder("data/snp_50_stocks_full")

    train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
    train_episodes = total_training_steps // (len(train_dfs[0]) - 1)
    test_dfs = [s.df.loc[1000:] for s in stocks[:]]

    # Agents
    # DQN Untrained
    testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
    for i in range(100): # Increased
        train_env = PortfolioAllocationEnvironment(train_dfs, attributes, discrete=True)
        train_env.reset()
        model = DQN('MlpPolicy', train_env, verbose=0, learning_rate=0.0001, gamma=0)
        test_env = PortfolioAllocationEnvironment(train_dfs, attributes, discrete=True)
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
    testing_results.to_excel(path + f"DQN Untrained_testing.xlsx")
    
    # DQN
    training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
    testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
    for i in range(repeats):
        print(f"Repeat {i + 1}")
        train_env = PortfolioAllocationEnvironment(train_dfs, attributes, discrete=True)
        train_env.reset()
        model = DQN('MlpPolicy', train_env, verbose=0, learning_rate=0.0001, gamma=0)
        model.learn(total_timesteps=total_training_steps)

        training_results[i + 1] = train_env.final_values

        test_env = PortfolioAllocationEnvironment(test_dfs, attributes, discrete=True)
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break

        testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
    seperator = "-"        
    training_results.to_excel(path + f"DQN_training.xlsx")
    testing_results.to_excel(path + f"DQN_testing.xlsx")

    # A2C Untrained
    testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
    for i in range(repeats):
        train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=0.0005, gamma=0)
        test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
    testing_results.to_excel(path + f"A2C Untrained_testing.xlsx")
    
    # A2C
    training_results = pd.DataFrame({"Episode": [(i + 1) for i in range(train_episodes)]})
    testing_results = pd.DataFrame({"Values": ["Final Value", "Annualized Return", "Sharpe Ratio"]})
    for i in range(repeats):
        print(f"Repeat {i + 1}")
        train_env = PortfolioAllocationEnvironment(train_dfs, attributes)
        train_env.reset()
        model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=0.0005, gamma=0)
        model.learn(total_timesteps=total_training_steps)

        training_results[i + 1] = train_env.final_values

        test_env = PortfolioAllocationEnvironment(test_dfs, attributes)
        obs = test_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if done:
                break
        testing_results[i + 1] = [test_env.portfolio_value, test_env.annualized_return, test_env.sharpe_ratio]
    seperator = "-"        
    training_results.to_excel(path + f"A2C_training.xlsx")
    testing_results.to_excel(path + f"A2C_testing.xlsx")
