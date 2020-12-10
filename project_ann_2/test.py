import numpy as np
import matplotlib.pyplot as plt
import torch
from env import Env
from agent import Agent


n_epsidoes = 1000
lr = 0.001
gamma = 0.99
ann_layers = [200, 100]
memory_size = 2000
batch_size = 100
eps_start = 1.00
eps_min = 0.01
eps_dec = 5e-6
replace_every = 500

env = Env("./data/stocks_train.csv", 10000, 15)
agent = Agent(env, gamma, lr, env.n_actions, env.state_dim, ann_layers, memory_size, batch_size, eps_start, eps_min, eps_dec, replace_every)
state = env.reset()

scores = []
for ep in range(n_epsidoes):
    done = False
    state = env.reset()
    earned = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, done, next_state)
        state = next_state
        agent.learn()
        earned += reward.item()
    scores.append(earned)
    
    print(f'Train :: Episode [{ep+1}/{n_epsidoes}]. Earned:{earned:.3f}. Loss : {agent.last_loss:.3f}. Epsilon : {agent.epsilon:.3f}')