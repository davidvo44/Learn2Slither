import torch.nn as nn
import torch
import torch.optim as optim


# class DQN(nn.Module):
#     def __init__(self, input_dim, n_action, lr):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(*input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, n_action)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.loss = nn.MSELoss()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
    

class DQN(nn.Module):
    def __init__(self, input_dim, n_action, lr):
        super(DQN, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
                                     nn.Linear(512, n_action),)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.network(x / 255.)

'''Conv2d
Conv2d
Conv2d
Flatten
Linear
Linear'''
