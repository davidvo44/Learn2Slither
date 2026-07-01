
class HyperParameters():
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0,
                 epislon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=1000,
                 memory_size=1000, episodes=1000):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epislon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.episodes = episodes
