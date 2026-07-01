import torch.nn as nn
import torch
import torch.optim as optim
from torchrl.data import ReplayBuffer


class DDQNAgent(object):
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0,
                 epislon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=1000,
                 memory_size=1000, episodes=1000):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epislon_min
        self.eps_dec = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.episodes = episodes

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir