import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, n_action, lr):
        super(DQN, self).__init__()
        self.network = nn.Sequential(nn.Linear(*input_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, n_action))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x):
        return self.network(x)
