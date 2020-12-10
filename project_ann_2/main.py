import numpy as np
import matplotlib.pyplot as plt
from env import Env
from agent import Agent

n_epsidoes = 1000

env = Env("./data/stocks_train.csv", 10000, 15)
state = env.reset()

agent = Agent(env=env, gamma=.999, lr=.002, n_actions=env.n_actions, input_dim=env.state_dim, 
               no_rnn_hidden=100, no_rnn_layer=1, ann_layer=[516], mem_size=500, batch_size=100, epsilon=1.0, 
                eps_min=0.01, eps_dec=5e-8, replace=500, path='tmp')

env_test = Env("./data/stocks_test.csv", 10000, 15)
scores = []
scores_ = []
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
    
    print(f'Train :: Episode [{ep+1}/{n_epsidoes}]. Earned:{earned:.3f}. Loss : {agent.last_loss:.3f}.')

    state_ = env_test.reset()
    earned_ = 0
    done_ = False
    while not done_:
        action = agent.choose_action(state_)
        next_state, reward, done_, info = env_test.step(action)
        agent.store_transition(state_, action, reward, done, next_state)
        state_ = next_state
        earned_ += reward.item()
    print(f'Test :: Episode [{ep+1}/{n_epsidoes}]. Earned:{earned_:.3f}.')
    scores_.append(earned_)

plt.plot(scores)
plt.title("Train Data")
plt.show()

plt.plot(scores_)
plt.title("Test Data")
plt.show()

