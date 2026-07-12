import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from ddqnAgent import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt


def train_ddqn():
    print('hey')
    env = gym.make("CartPole-v1",
                   render_mode="rgb_array")
    n_games = 2000
    training_period = 100

    env = RecordVideo(
        env,
        video_folder='CartPole-agent',
        name_prefix='train',
        episode_trigger=lambda x: x % training_period == 0
    )

    env = RecordEpisodeStatistics(env)
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    agent = DDQNAgent(n_actions=output_dim, input_dims=input_dim)
    scores = []
    steps_array = []
    eps_history = []
    avg_score = []

    best_score = 0.0
    load_checkpoint = False
    scores = []
    for episode in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        if episode % 100 == 0:
            print(episode)
        n_steps = 0
        while not done:
            action = agent.choose_action(observation, env)
            step_result = env.step(action)

            if len(step_result) == 4:
                observation_, reward, done, _ = step_result
            else:
                observation_, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            score += reward

            if not load_checkpoint:
                agent.memory.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        if avg_score[-1] > best_score:
            best_score = avg_score[-1]
        avg_score.append(np.mean(scores[-100:]))
        print('episode:', episode, 'score:', score,
              'average score %.1f' % avg_score[-1],
              'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        
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
