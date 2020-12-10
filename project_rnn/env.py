import torch
import numpy as np
import pandas as pd
import itertools

class Env(object):
    def __init__(self, filename, initial_amount, history_n):
        df = pd.read_csv(filename)
        data = df.values
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N, self.D = data.shape
        self.num_iters = 0
        self.curr = initial_amount
        self.initial_amount = initial_amount
        self.raw_data = torch.tensor(data, dtype=torch.float, device=self.device)
        self.history_n = history_n

        # actions
        # 0 - buy
        # 1 - sell
        # 2 - stay
        self.n_actions = self.D ** 3
        self.action_space = list(map(list, itertools.product([0, 1, 2], repeat=self.D)))
        # states
        self.state_dim = self.D * 2 + 1
        self.stock_owned = torch.zeros(self.D, dtype=torch.long, device=self.device)
        
        self.state_history = torch.zeros(self.history_n, self.state_dim, device=self.device)

        self.initialize_scaler()
        self.reset()

    def _reset_(self):
        self.asset = self.initial_amount
        self.position = 0
        self.curr_state = torch.zeros(self.D, device=self.device)
        self.stock_price = self.raw_data[self.position]
        return self._get_obs_()

    def reset(self):
        self.asset = self.initial_amount
        self.position = 0
        self.curr_state = torch.zeros(self.D, device=self.device)
        self.stock_price = self.raw_data[self.position]
        self.state_history[-1] = self._get_obs_()
        return self.state_history.clone()
    
    # Returns state scaler and reward scaler
    # 0 index - mean
    # 1 index - std
    def initialize_scaler(self):
        state = self._reset_()
        states = torch.zeros(self.N, state.shape[0], device=self.device)
        self.state_scaler = torch.zeros(self.state_dim, 2, device=self.device)
        self.reward_scaler = torch.zeros(2, device=self.device)
        rewards = torch.zeros(self.N, device=self.device)
        states[0] = state
        rewards[0] = 0
        for step in range(self.N - 1):
            step = step + 1
            random_action = np.random.choice(self.n_actions)
            next_state, reward, _, _ = self._step_(random_action)
            states[step] = state
            rewards[step] = reward
            state = next_state

        # State scaler
        for col in range(self.state_dim):
            mean, std = torch.mean(states[:,col]), torch.std(states[:, col])
            self.state_scaler[col][0] = mean
            self.state_scaler[col][1] = std
        
        # Reward scaler
        mean, std = torch.mean(rewards), torch.std(rewards)
        self.reward_scaler[0] = mean
        self.reward_scaler[1] = std
        self.reset()
        return self.state_scaler, self.reward_scaler

    def transform_state(self, state):
        state_copy = state.clone()
        state_copy = (state_copy - self.state_scaler[:, 0]) / self.state_scaler[:, 1]
        return state_copy

    def transform_reward(self, reward):
        reward_copy = reward.clone()
        reward_copy = (reward_copy - self.reward_scaler[0]) / self.reward_scaler[1]
        return reward_copy

    def step(self, action):
        prev_value = self._get_value_()
        
        self.num_iters += 1
        self.position += 1
        self.stock_price = self.raw_data[self.position]
        self._trade_(action)

        curr_value = self._get_value_()
        reward = curr_value - prev_value
        done = (self.position + 1) == self.N
        info = {'curr_value' : curr_value }
        self.state_history[:-1] = self.state_history[1:].clone()
        self.state_history[-1] = self._get_obs_().clone()
        return self.state_history.clone(), reward, done, info

    def _step_(self, action):
        prev_value = self._get_value_()
        
        self.num_iters += 1
        self.position += 1
        self.stock_price = self.raw_data[self.position]
        self._trade_(action)

        curr_value = self._get_value_()
        reward = curr_value - prev_value
        done = (self.position + 1) == self.N
        info = {'curr_value' : curr_value }
        
        return self._get_obs_(), reward, done, info

    def _get_obs_(self):
        obs = torch.zeros(self.state_dim, device=self.device)
        obs[:self.D] = self.stock_owned
        obs[self.D:self.D * 2] = self.stock_price
        obs[-1] = self.asset
        return obs

    def _get_value_(self):
        return self.stock_price.dot(self.stock_owned.float()) + self.asset

    def _trade_(self, action):
        given_action = self.action_space[action]
        stocks_to_buy = []
        stocks_to_sell = []
        for stock, action in enumerate(given_action):
            if action == 0:
                stocks_to_buy.append(stock)
            elif action == 1:
                stocks_to_sell.append(stock)

        if len(stocks_to_buy) > 0:
            stock_to_buy_index = 0
            stock_to_buy = stocks_to_buy[stock_to_buy_index]
            while self.stock_price[stock_to_buy] < self.asset:
                self.stock_owned[stock_to_buy] += 1
                self.asset -= self.stock_price[stock_to_buy]
                stock_to_buy_index = (stock_to_buy_index + 1) % len(stocks_to_buy)
                stock_to_buy = stocks_to_buy[stock_to_buy_index]

        if len(stocks_to_sell) > 0:
            for target_stock in stocks_to_sell:
                if self.stock_owned[target_stock] > 0:
                    num_to_sell = self.stock_owned[target_stock].clone()
                    self.stock_owned[target_stock] = 0
                    self.asset += self.stock_price[target_stock] * num_to_sell
