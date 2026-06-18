import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np

if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        map_name='8x8',
        is_slippery=True,
        render_mode="human"
    )

    q = np.zeros((env.observation_space.n, env.action_space.n))

    lr = 0.9
    discount_factor_g = 0.9

    state = env.reset()[0]
    done = False

    while not done:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)

        env.render()   # 👈 affichage

        done = terminated or truncated

    env.close()
