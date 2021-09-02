# Overview

The key features of this repository are listed below.

- Class for stock data extraction, processing and storage.
- Portfolio allocation environment for deep reinforcement learning with either discrete or continuous action spaces.
- Dataset for the top 50 S&P 500 stocks including price data, volumes data and news headlines; as well as technical indicators and sentiment scores derived from this data.

# Installation

To install the project using pip with python the following steps should be followed.

## Creating Virtual Environment

First pip install virtualenv by running:

```bash
pip install virtualenv
```

Next, while in the folder for this project create a new virtual environment using:

```bash
python -m venv venv
```

## Activating Virtual Environment

From the project folder the virtual environment can now be activated. If using Windows this can be done by running:

```bash
venv\Scripts\activate
```
Or, if using Linux or Mac running:

```bash
source venv/bin/activate
```

## Installing Dependencies

Finally, with this new environment activated the dependencies for the project can be installed by running:

```bash
pip install -r requirements.txt
```

## Downloading Chrome Web Driver (Optional)
This step is required in order to extract news headlines on given stocks, which is a part of the `extract_and_calculate_all` method but is not included in the `extract_and_calculate_basic` method.

For this to work Chrome will already need to be installed. If this is the case then go to https://chromedriver.chromium.org/downloads and download the relevant file for your operating system and version of Chrome. Put this downloaded file in the "other\chromedriver" folder. When run the code should recognise this file, providing its filename starts with "chromedriver"

# Usage

To run files from this repository first activate the virtual environment as described in the installation.

## Data Extraction and Processing

To extract and process data first create a `Stock` object in the `main.py` file. Data can then be extracted from the given stock and processed using one of the extraction methods such as `extract_and_calculate_basic`. To get the DataFrame from this object the attribute can simply be assigned to a variable. An example of this process is provided below.

```python
s = Stock('Apple', 'AAPL')
s.extract_and_calculate_basic()
df = s.df
```

## Data Storage and Retrieval

The `storage.py` file includes some functions for saving and loading stock objects to and from dill files. To save and retrieve stocks the `save_dill_object` and `retrieve_dill_object` functions can be used. For example.

```python
save_dill_object(s, 'data/my_stock.dill')
retrived_stock = retrieve_dill_object('data/my_stock.dill')
```
Another function has also been provided to retrieve all stocks in a given folder called `retrieve_stocks_from_folder`. An example of using this function to extract the provided dataset, the top 50 S&P 500 stocks, is given below.

```python
list_of_stocks = retrieve_stocks_from_folder('data/snp_50_stocks_full')
```

## Portfolio Allocation Environment

A portfolio allocation environment for reinforcement learning can be created by instantiating an object of the `PortfolioAllocationEnvironment` class. The constructor of this class takes three main parameters. `stocks` should be provided as a list of stock DataFrames, which should be of the same length and over the same time period. `state_attributes` takes a list of attributes, as strings, that will be used to represent the state space. To decide which of these to use check the column names of the stock DataFrames. The final parameter is the boolean `discrete`, which determines how the action space will be structured. If this is set to true the action space is discrete, otherwise it will be continuous. A full example of creating an environment is provided below.

```python
stocks = retrieve_stocks_from_folder("data/snp_50_stocks_full")
dfs = [s.df for s in stocks]
attributes = ['normalized_rsi', 'std_devs_out', 'relative_vol', 'hf_google_articles_score', 'vader_google_articles_score', 'ranking_score']

env = PortfolioAllocationEnvironment(dfs, attributes, discrete=True)
```

## Using Environment

To use this environment with a deep reinforcement learning agent the data should be split into a testing and training set and two different environments can be constructed. An agent can then be trained on the first environment and tested on the second environment. An example of this process is provided below using an agent from Stable Baselines 3.

```python
from stable_baselines3 import A2C

# Create and Train Agent
attributes = ['std_devs_out', 'vader_google_articles_score']
train_dfs = [s.df.loc[100:1000] for s in stocks[:]]
train_env = PortfolioAllocationEnvironment(
        train_dfs, attributes, discrete=False)
model = A2C('MlpPolicy', train_env, verbose=0, learning_rate=0.0005, gamma=0)
model.learn(total_timesteps=100_000)

# Test Agent
test_dfs = [s.df.loc[1000:] for s in stocks[:]]
test_env = PortfolioAllocationEnvironment(
        test_dfs, attributes, discrete=False)
obs = test_env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    if done:
        break
print(test_env.annualized_return)
```
