from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackJackAgent:
    def __init__(self,
                 env: gym.Env,
                 learningRate: float,
                 initialEpsilon: float,
                 epsilonDecay: float,
                 finalEpsilon: float,
                 discountFactor: float = 0.95):
        self.env = env
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learningRate
        self.epsilon = initialEpsilon
        self.epsilonDecay = epsilonDecay
        self.finalEpsilon = finalEpsilon
        self.discountFactor = discountFactor  # How much we care about future rewards

        self.trainingError = []

    def getAction(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.qValues[obs]))
        
    def update(self,
               obs: tuple[int, int, bool],
               action: int,
               reward: float,
               terminated: bool,
               nextObs: tuple[int, int, bool]):
    
        futureQ = (not terminated) * np.max(self.qValues[nextObs])
        target = reward + self.discountFactor * futureQ
        temporalDifference = target - self.qValues[obs][action]

        self.qValues[obs][action] = (
            self.qValues[obs][action] + self.lr * temporalDifference
        )

        self.trainingError.append(temporalDifference)

    def decayEpsilon(self):
        self.epsilon = max(self.finalEpsilon, self.epsilon - self.epsilonDecay)

