import torch.nn as nn
import torch
import torch.optim as optim



class DQN(nn.Module):
    def __init__(self, input_dim, n_action, lr):
        super(DQN, self).__init__()
        self.network = nn.Sequential(nn.Linear(*input_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, n_action), nn.ReLU(),)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()  #  PREVOIR DE LE CHANGER

    def forward(self, x):
        return self.network(x)
