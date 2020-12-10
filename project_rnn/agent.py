import numpy as np
import torch
import torch.nn as nn

from dqn import DQN
from replay_memory import ReplayMemory

class Agent():
    def __init__(self, env, gamma, lr, n_actions, input_dim, no_rnn_hidden, no_rnn_layer, ann_layer, 
                mem_size, batch_size, epsilon, 
                eps_min=0.01, eps_dec=5e-6, replace=500, path='tmp'):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.path = path
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size)

        self.q_eval = DQN(input_dim, no_rnn_hidden, no_rnn_layer, ann_layer, n_actions, self.batch_size)
        self.q_next = DQN(input_dim, no_rnn_hidden, no_rnn_layer, ann_layer, n_actions, self.batch_size)

        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        self.last_loss = 0

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = self.env.transform_state(state)
            self.q_eval.eval()
            with torch.no_grad():
                actions = self.q_eval(state.reshape(1, state.shape[0], state.shape[1]))
            self.q_eval.train()
            return actions.argmax().item()
        else:
            return np.random.choice(self.action_space)

    def store_transition(self, state, action, reward, done, next_state):
        self.memory.push(state.reshape(-1, state.shape[0], state.shape[1]), action, reward, done, next_state.reshape(-1, state.shape[0], state.shape[1]))

    def sample_memory(self):
        state, action, reward, done, next_state = self.memory.sample(self.batch_size)
        device = self.q_eval.device
        state, action, reward, done, next_state = \
                             state.to(device), action.to(device), reward.to(device), done.to(device), next_state.to(device)
        return state, action, reward, done, next_state
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon - self.eps_dec > self.eps_min \
                            else self.eps_min

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, done, next_state = self.sample_memory()
        state = self.env.transform_state(state)
        reward = self.env.transform_reward(reward)
        next_state = self.env.transform_state(next_state)
        q_pred = self.q_eval(state)
        q_pred = q_pred[torch.arange(self.batch_size), action.long()]
        q_next = self.q_next(next_state).max(1)[0]

        q_next[done] = 0.0
        q_target = reward + self.gamma * q_next
        
        loss = self.loss(q_pred, q_target.detach()).to(self.q_eval.device)
        self.last_loss = loss.item()
        loss.backward()
        
        self.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save(self, path):
        self.q_eval.save_model(self.path)

    def load(self, path):
        self.q_eval.load_model(self.path)
        self.q_next.load_model(self.path)