# Old experiments

def experiment_2():
    path = "data/results/experiment_2_raw/"
    test_attributes = [
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol'],
        ['ranking_change_score'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol', 'ranking_change_score'],
        ['macd'],
        ['signal_line'],
        ['normalized_rsi'],
        ['std_devs_out'],
    ]

    test_attributes = [
        ['relative_vol'],
        ['ranking_score'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol', 'ranking_score'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol', 'ranking_change_score', 'ranking_score']
    ]
    test_attributes = []

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

def experiment_3():
    path = "data/results/experiment_3_raw/"
    test_attributes = [
        ['hf_google_articles_score'],
        ['ranking_change_score', 'ranking_score', 'hf_google_articles_score'],
        ['macd', 'signal_line', 'normalized_rsi', 'std_devs_out', 'relative_vol', 'ranking_change_score', 'ranking_score', 'hf_google_articles_score']   
    ]
    test_attributes = []

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
