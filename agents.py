"""
This file should contain a number of agent classes
"""

from portfolio import *
from environments import *
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import numpy as np

from collections import deque 

# class BenchmarkAgent:
#     def __init__(self, portfolio):
#         self.portfolio = portfolio
#         self.values = []

#     def make_action(self):
#         self.portfolio.buy_all_equally()
#         value, _ = self.portfolio.action([], [])
#         self.values.append(value)

#     def run_episode(self):
#         for i in range(253):
#             self.make_action()


# class SplitAndHoldAgent:
#     def __init__(self, portfolio):
#         self.portfolio = portfolio
#         self.values = []

#     def make_action(self):
#         value, _ = self.portfolio.action([], [])
#         self.values.append(value)

#     def run_episode(self):
#         for i in range(253):
#             self.make_action()

class SingleStockDQNAgentWithDNN:
    def __init__(self):
        self.action_rep =  {0: "Buy", 1: "Sell", 2: "Hold"}

        # Parameters
        self.gamma = 0.8
        self.epsilon = 0.1
        self.alpha = 0.01
        self.c = 100
        self.d_min = 100
        self.d_max = 100000
        self.number_of_replays = 10

        self.env = SingleStockEnvironment()
        self.network = self._build_DNN()
        self.target_network = self._build_DNN()
        self.target_network.set_weights(self.network.get_weights())
        self.replay_memory = deque([])
        

    def _build_DNN(self):
        model = Sequential([
            Dense(units=16, input_shape=(6,), activation='relu'),
            Dense(units=16, activation='relu'),
            Dense(units=3, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.alpha))
        return model

    def _get_states_action_values_from_network(self, state, selected_network):
        dnn_inputs = np.array(state).reshape(-1,6) # Transforms into array of states
        dnn_outputs = selected_network.predict(dnn_inputs) # Returns array of state action arrays
        return dnn_outputs[0]

    def _get_greedy_action(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.network)
        print(dnn_out)
        max_index = np.argmax(dnn_out)
        return max_index

    def _get_random_action(self):
        return random.choice(self.env.get_available_actions())

    def _get_epsilon_greedy_action(self, state):
        if self.epsilon > random.random():
            return self._get_random_action()
        else:
            return self._get_greedy_action(state)
    
    def _get_greedy_target_action_value(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.target_network)
        return max(dnn_out)

    def _replay(self, verbose=0):
        # Pick sample of transitions (state, action, reward, new_state, terminal)
        replays = random.sample(self.replay_memory, self.number_of_replays)
        # Calculate fit input values for these transitions
        xs = []
        ys = []
        new_state_values = self.target_network.predict(np.array([t[3] for t in replays]))
        state_values = self.network.predict(np.array([t[0] for t in replays]))
        for index, (state, action, reward, new_state, terminal) in enumerate(replays):
            if not terminal:
                fit_value = reward + (self.gamma * np.amax(new_state_values[index]))
            else:
                fit_value = reward
            qs = state_values[index]
            qs[action] = fit_value

            xs.append(state)
            ys.append(qs)
        # Perform gradient descent step
        self.network.fit(np.array(xs), np.array(ys), batch_size=self.number_of_replays, verbose=1, shuffle=False)

    def train(self, episodes=1, verbose=0):
        step = 1
        for e in range(episodes):
            print("Episode:", e)
            # Initialise initial state
            state, reward, terminal = self.env.reset()
            while not terminal:
                # Get epsilon greedy action
                action = self._get_epsilon_greedy_action(state)
                # Execute action
                new_state, reward, terminal = self.env.make_action(action)
                # Store observed transition
                if len(self.replay_memory) >= self.d_max:
                    self.replay_memory.popleft()
                self.replay_memory.append((state, action, reward, new_state, terminal))
                # Run replay batch
                if len(self.replay_memory) >= self.d_min:
                    self._replay(verbose=verbose)
                # Set target network to network every C steps
                if step % self.c == 0:
                    self.target_network.set_weights(self.network.get_weights())
                # Increment step counter
                state = new_state
                step += 1
            # benchmark_value = self.env.benchmark_portfolio.get_portfolio_value()
            # value = self.env.portfolio.get_portfolio_value()
            # print(value - benchmark_value)
            plt.plot(self.env.values, label="Agent")
            plt.plot(self.env.stock_values, label="Benchmark")
            plt.title(f"{e}-{self.env.stock.code}")
            plt.xlabel("Episodes")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.savefig(f"data/figs/{e}-{self.env.stock.code}")
            plt.clf()

class SingleStockWeightingDQNAgentWithDNN:
    def __init__(self):
        self.action_rep =  {0: "Buy", 1: "Sell", 2: "Hold"}
        # Parameters
        self.gamma = 0.8
        self.epsilon = 0.1
        self.alpha = 0.005
        self.c = 100
        self.d_min = 1000
        self.d_max = 100000
        self.number_of_replays = 10
        # Other attributes
        self.env = SingleStockWeightControlEnvironment()
        self.network = self._build_DNN()
        self.target_network = self._build_DNN()
        self.target_network.set_weights(self.network.get_weights())
        self.replay_memory = deque([])
        

    def _build_DNN(self):
        model = Sequential([
            Dense(units=16, input_shape=(6,), activation='relu'),
            Dense(units=16, activation='relu'),
            Dense(units=3, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.alpha))
        return model

    def _get_states_action_values_from_network(self, state, selected_network):
        dnn_inputs = np.array(state).reshape(-1,6) # Transforms into array of states
        dnn_outputs = selected_network.predict(dnn_inputs) # Returns array of state action arrays
        return dnn_outputs[0]

    def _get_greedy_action(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.network)
        # print(dnn_out)
        max_index = np.argmax(dnn_out)
        return max_index

    def _get_random_action(self):
        return random.choice(self.env.get_available_actions())

    def _get_epsilon_greedy_action(self, state):
        if self.epsilon > random.random():
            return self._get_random_action()
        else:
            return self._get_greedy_action(state)
    
    def _get_greedy_target_action_value(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.target_network)
        return max(dnn_out)

    def _replay(self, verbose=0):
        # Pick sample of transitions (state, action, reward, new_state, terminal)
        replays = random.sample(self.replay_memory, self.number_of_replays)
        # Calculate fit input values for these transitions
        xs = []
        ys = []
        new_state_values = self.target_network.predict(np.array([t[3] for t in replays]))
        state_values = self.network.predict(np.array([t[0] for t in replays]))
        for index, (state, action, reward, new_state, terminal) in enumerate(replays):
            if not terminal:
                fit_value = reward + (self.gamma * np.amax(new_state_values[index]))
            else:
                fit_value = reward
            qs = state_values[index]
            qs[action] = fit_value

            xs.append(state)
            ys.append(qs)
        # Perform gradient descent step
        self.network.fit(np.array(xs), np.array(ys), batch_size=self.number_of_replays, verbose=verbose, shuffle=False)

    def train(self, episodes=1, verbose=0):
        step = 1
        for e in range(episodes):
            print("Episode:", e)
            # Initialise initial state
            state, reward, terminal = self.env.reset()
            while not terminal:
                # Get epsilon greedy action
                action = self._get_epsilon_greedy_action(state)
                # Execute action
                new_state, reward, terminal = self.env.make_action(action)
                # Store observed transition
                if len(self.replay_memory) >= self.d_max:
                    self.replay_memory.popleft()
                self.replay_memory.append((state, action, reward, new_state, terminal))
                # Run replay batch
                if len(self.replay_memory) >= self.d_min:
                    self._replay(verbose=verbose)
                # Set target network to network every C steps
                if step % self.c == 0:
                    self.target_network.set_weights(self.network.get_weights())
                # Increment step counter
                state = new_state
                step += 1

            self.save_plot_of_episode_results(e)
            
    def save_plot_of_episode_results(self, episode):
        plt.plot(self.env.values, label="Agent")
        plt.plot(self.env.stock_values, label="Stock")
        plt.plot(self.env.market_values, label="Market")
        year_index = str(int((self.env.year_indices[0] - 59) / 253))
        title = f"{episode}-{self.env.stock.code}-Y{year_index}"
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.savefig(f"data/figs/{title}")
        plt.clf()

class V2SingleStockWeightingDQNAgentWithDNN:
    """Replay moved to between episodes as a way to have larger batches"""
    def __init__(self):
        self.action_rep =  {0: "Buy", 1: "Sell", 2: "Hold"}
        # Parameters
        self.gamma = 0.8
        self.epsilon = 0.1
        self.alpha = 0.005
        self.c = 100
        self.d_min = 2000
        self.d_max = 100000
        self.number_of_replays = 1000
        # Other attributes
        self.env = SingleStockWeightControlEnvironment()
        self.network = self._build_DNN()
        self.target_network = self._build_DNN()
        self.target_network.set_weights(self.network.get_weights())
        self.replay_memory = deque([])
        

    def _build_DNN(self):
        model = Sequential([
            Dense(units=16, input_shape=(6,), activation='relu'),
            Dense(units=16, activation='relu'),
            Dense(units=3, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.alpha))
        return model

    def _get_states_action_values_from_network(self, state, selected_network):
        dnn_inputs = np.array(state).reshape(-1,6) # Transforms into array of states
        dnn_outputs = selected_network.predict(dnn_inputs) # Returns array of state action arrays
        return dnn_outputs[0]

    def _get_greedy_action(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.network)
        # print(dnn_out)
        max_index = np.argmax(dnn_out)
        return max_index

    def _get_random_action(self):
        return random.choice(self.env.get_available_actions())

    def _get_epsilon_greedy_action(self, state):
        if self.epsilon > random.random():
            return self._get_random_action()
        else:
            return self._get_greedy_action(state)
    
    def _get_greedy_target_action_value(self, state):
        dnn_out = self._get_states_action_values_from_network(state, self.target_network)
        return max(dnn_out)

    def _replay(self, verbose=0):
        # Pick sample of transitions (state, action, reward, new_state, terminal)
        replays = random.sample(self.replay_memory, self.number_of_replays)
        # Calculate fit input values for these transitions
        xs = []
        ys = []
        new_state_values = self.target_network.predict(np.array([t[3] for t in replays]))
        state_values = self.network.predict(np.array([t[0] for t in replays]))
        for index, (state, action, reward, new_state, terminal) in enumerate(replays):
            if not terminal:
                fit_value = reward + (self.gamma * np.amax(new_state_values[index]))
            else:
                fit_value = reward
            qs = state_values[index]
            qs[action] = fit_value

            xs.append(state)
            ys.append(qs)
        # Perform gradient descent step
        self.network.fit(np.array(xs), np.array(ys), batch_size=32, verbose=verbose, shuffle=False)

    def train(self, episodes=1, verbose=0, save_figs=False):
        step = 1
        for e in range(episodes):
            print("Episode:", e)
            # Initialise initial state
            state, reward, terminal = self.env.reset()
            while not terminal:
                # Get epsilon greedy action
                action = self._get_epsilon_greedy_action(state)
                # Execute action
                new_state, reward, terminal = self.env.make_action(action)
                # Store observed transition
                if len(self.replay_memory) >= self.d_max:
                    self.replay_memory.popleft()
                self.replay_memory.append((state, action, reward, new_state, terminal))
                
                # Set target network to network every C steps
                if step % self.c == 0:
                    self.target_network.set_weights(self.network.get_weights())
                # Increment step counter
                state = new_state
                step += 1
            # Run replay batch
            if len(self.replay_memory) >= self.d_min:
                self._replay(verbose=verbose)

            if save_figs: self.save_plot_of_episode_results(e)
            
    def save_plot_of_episode_results(self, episode):
        plt.plot(self.env.values, label="Agent")
        plt.plot(self.env.stock_values, label="Stock")
        plt.plot(self.env.market_values, label="Market")
        year_index = str(int((self.env.year_indices[0] - 59) / 253))
        title = f"{episode}-{self.env.stock.code}-Y{year_index}"
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.savefig(f"data/figs/{title}")
        plt.clf()

if __name__ == "__main__":
    agent = V2SingleStockWeightingDQNAgentWithDNN()
    agent.train(100, verbose=0)