import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation

if __name__ == "__main__":
    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        size=10,
        render_mode="human"
    )
    obs, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()   # 👈 affichage

        done = terminated or truncated

    env.close()
