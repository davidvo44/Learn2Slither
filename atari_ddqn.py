import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from ddqnAgent import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import ale_py


def atari_ddqn():
    gym.register_envs(ale_py)
    n_games = 5000
    training_period = 100

    env = gym.make('ALE/Breakout-v5',
                   render_mode="rgb_array",
                   frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=30,
                                          frame_skip=4,
                                          screen_size=84,
                                          grayscale_obs=True,
                                          terminal_on_life_loss=False)
    
    env = gym.wrappers.FrameStackObservation(env, 4)

    env = RecordVideo(
        env,
        video_folder='AtariGame-agent',
        name_prefix='train',
        episode_trigger=lambda x: x % training_period == 0
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)

    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    print(input_dim)
    agent = DDQNAgent(n_actions=output_dim, input_dims=input_dim)
    scores = []
    steps_array = []
    eps_history = []
    avg_score = []

    best_score = 0.0
    load_checkpoint = False
    for episode in range(n_games):
        score = 0
        observation, _ = env.reset()
        observation, reward, terminated, truncated, info = env.step(1)
        clipped_reward = np.sign(reward)
        score += clipped_reward
        current_lives = info.get('lives', 5)
        done = terminated or truncated
        if episode % 100 == 0:
            print(episode)
        n_steps = 0
        while not done:
            ram_before = env.unwrapped.ale.getRAM().copy()
            action = agent.choose_action(observation, env)

            observation_, reward, terminated, truncated, info = env.step(action)
            print('SHAPE',)
            clipped_reward = np.sign(reward)

            done = terminated or truncated

            new_lives = info.get('lives', 5)
            if new_lives < current_lives and not (terminated or truncated):
                next_obs, fire_reward, terminated, truncated, info = env.step(1)
                clipped_reward += fire_reward  # On cumule la récompense si besoin
                current_lives = new_lives
            else:
                current_lives = new_lives
            ram_after = env.unwrapped.ale.getRAM().copy()
            # if ram_after[4] > ram_before[4]:
            #     score += 1
            # if ram_after[4] < ram_before[4]:
            #     score -= 1
            score += clipped_reward

            if not load_checkpoint:
                agent.memory.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score.append(np.mean(scores[-100:]))
        print('episode:', episode, 'score:', score,
              'average score %.1f' % avg_score[-1],
              'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score[-1] > best_score:
            best_score = avg_score[-1]
        
        eps_history.append(agent.epsilon)

        if load_checkpoint and n_steps >= 18000:
            break
    env.close()
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(avg_score)
    plt.ylabel("Average score")

    plt.subplot(2, 1, 2)
    plt.plot(eps_history)
    plt.ylabel("Epsilon")
    plt.xlabel("Episode")

    plt.savefig("dqncartpole.png")
