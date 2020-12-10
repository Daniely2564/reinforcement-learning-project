import random
import torch
import numpy as np

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {
            'state':[],
            'action':[],
            'reward':[],
            'done':[],
            'next_state':[]
        }
        self.position = 0

    def push(self, state, action, reward, done, next_state):
        """Saves a transition."""
        if len(self.memory['state']) < self.capacity:
            self._add_none()
        self._append_to_memory(state, action, reward, done, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ran_idx = np.random.choice(self.size(), batch_size, replace=False)
        state = torch.cat(self.memory['state'])[ran_idx]
        action = torch.tensor(self.memory['action'])[ran_idx]
        reward = torch.tensor(self.memory['reward'])[ran_idx]
        done = torch.tensor(self.memory['done'])[ran_idx]
        next_state = torch.cat(self.memory['next_state'])[ran_idx]
        return state, action, reward, done, next_state

    def _add_none(self):
        self.memory['state'].append(None)
        self.memory['action'].append(None)
        self.memory['reward'].append(None)
        self.memory['done'].append(None)
        self.memory['next_state'].append(None)
    
    def _append_to_memory(self, state, action, reward, done, next_state):
        self.memory['state'][self.position] = state
        self.memory['action'][self.position] = action
        self.memory['reward'][self.position] = reward
        self.memory['done'][self.position] = done
        self.memory['next_state'][self.position] = next_state

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.memory['state'])