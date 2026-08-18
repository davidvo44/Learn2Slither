import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from src.ddqn_Agent_Snake import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt


def learnSnake(hp, filepath=None):
    os.makedirs(os.path.dirname('src/learning_data/file'), exist_ok=True)
    n_games = hp['episode_train']
    training_period = 200
    load_checkpoint = False

    scores = []
    eps_history = []
    avg_score = []
    best_score = -100

    if hp["visual"] is False:
        env = gym.make('gymnasium_env/GridWorld-v0', render_mode='rgb_array',
                       size=10)

        env = RecordVideo(
            env,
            video_folder='Snake-agent',
            name_prefix='train',
            episode_trigger=lambda x: x % training_period == 0
        )

        env = RecordEpisodeStatistics(env)
    else:
        env = gym.make('gymnasium_env/GridWorld-v0', render_mode='human',
                       size=10)

    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    agent = DDQNAgent(n_actions=output_dim, input_dims=input_dim, hp=hp)
    if filepath is not None:
        agent.load_model(filepath)
        agent.epsilon = 0.05
    for episode in range(n_games):
        score = 0
        observation, _ = env.reset()
        done = 0
        n_steps = 0
        while not done:
            if avg_score and avg_score[-1] > 30:
                action = agent.choose_action(observation, env)
            else:
                action = agent.choose_action(observation, env, mode='USE')
            observation_, reward, terminated, truncated, info =\
                env.step(action)
            done = terminated or truncated
            if hp["step_by_step"]:
                input('press enter to continue...')
            score += reward
            if not load_checkpoint:
                agent.memory.store_transition(observation, action,
                                              reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score.append(np.mean(scores[-100:]))

        print('episode:', episode,
              'average score %.1f' % avg_score[-1],
              'score: %.2f' % score,
              'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if score > best_score:
            best_score = score
            agent.save_model("src/learning_data/best_snake_model.pth")
        if episode % 200 == 0:
            agent.save_model(f'src/learning_data/checkpoint_ep_{episode}.pth')

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
