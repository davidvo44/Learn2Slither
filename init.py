import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        map_name='8x8',
        is_slippery=True,
        render_mode="rgb_array"
    )

    q = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00002     # epsilon decay rate. 1/0.0001 = 10,000
    epoch = 50000
    lr = 0.8
    discount_factor_g = 0.99
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(epoch)

    for i in range(epoch):
        state = env.reset()[0]
        done = False

        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] = q[state, action] + lr * (
                reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
            state = new_state
            done = terminated or truncated
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        lr = max(lr - 0.000005, 0.0001)

        if reward == 1:
            print(i)
            rewards_per_episode[i] = 1
    env.close()

    print(q)
    sum_rewards = np.zeros(epoch)
    for t in range(epoch):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen.png')

    viewer = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=True,
        render_mode="human"
    )

    state, _ = viewer.reset()
    done = False
    while not done:
        action = np.argmax(q[state])
        state, reward, terminated, truncated, _ = viewer.step(action)
        done = terminated or truncated
            
