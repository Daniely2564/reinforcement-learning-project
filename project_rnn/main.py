import numpy as np
import matplotlib.pyplot as plt
from env import Env
from agent import Agent

n_epsidoes = 1000

env = Env("./data/stocks_train.csv", 10000, 15)
state = env.reset()

# HYPERPARAMETERS
lr = 0.002
no_rnn_hidden = 100
no_rnn_layer = 1
ann_layer = [516]
memory_size = 500
batch_size = 100
eps_start = 1.0
eps_min = 0.01
eps_dec=5e-6
replace_target_model_every = 500

agent = Agent(env=env, gamma=.999, lr=lr, n_actions=env.n_actions, input_dim=env.state_dim, 
               no_rnn_hidden=no_rnn_hidden, no_rnn_layer=no_rnn_layer, ann_layer=ann_layer, mem_size=memory_size, batch_size=batch_size, epsilon=eps_start, 
                eps_min=eps_min, eps_dec=eps_dec, replace=replace_target_model_every, path='tmp')

env_test = Env("./data/stocks_test.csv", 10000, 15)

scores = []
scores_ = []
losses = []
epsilon_changes = []

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
    losses.append(agent.last_loss)
    epsilon_changes.append(agent.epsilon)

    print(f'Train :: Episode [{ep+1}/{n_epsidoes}]. Earned: {earned:12.3f}. Loss: {agent.last_loss:.3f}. Epsilon: {agent.epsilon:.3f}')

    state_ = env_test.reset()
    earned_ = 0
    done_ = False
    while not done_:
        action = agent.choose_action(state_)
        next_state, reward, done_, info = env_test.step(action)
        state_ = next_state
        earned_ += reward.item()
    print(f'Test  :: Episode [{ep+1}/{n_epsidoes}]. Earned: {earned_:12.3f}.')
    scores_.append(earned_)