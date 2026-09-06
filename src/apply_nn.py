import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from src.ddqn_Agent_Snake import DDQNAgent
import numpy as np
import gymnasium_env
from src.cli_snake_vision import cli_snake_vision


def use(hp, filepath=None, learn=False):
    n_games = hp['episode_apply']
    record_period = 50

    scores = []
    avg_score = []
    avg_body = []
    best_score = -100
    (gymnasium_env)
    if hp["visual"] is False:
        env = gym.make('gymnasium_env/GridWorld-v0',
                       render_mode='rgb_array', size=hp["board_size"])
        env = RecordVideo(
            env,
            video_folder='Snake-play',
            name_prefix='play',
            episode_trigger=lambda x: x % record_period == 0
        )

        env = RecordEpisodeStatistics(env)
    else:
        env = gym.make('gymnasium_env/GridWorld-v0',
                       render_mode='human', size=hp["board_size"])

    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    agent = DDQNAgent(hp=hp, n_actions=output_dim, input_dims=input_dim)
    if filepath is not None:
        agent.load_model(filepath)
    for episode in range(n_games):
        score = 0
        observation, _ = env.reset()
        done = 0
        n_steps = 0
        while not done:
            action = agent.choose_action(observation, env, visual=hp['visual'])
            if hp['step_by_step']:
                input('press enter to continue...')
            observation_, reward, terminated, truncated, info =\
                env.step(action)
            if hp["visual"]:
                cli_snake_vision(observation_)
            done = terminated or truncated
            score += reward
            observation = observation_
            n_steps += 1

        scores.append(score)
        avg_body.append(info['nbr_snake'])
        avg_score.append(np.mean(scores[-100:]))

        if not learn:
            print('episode:', episode,
                  'average score %.1f' % avg_score[-1],
                  'score: %.2f' % score,
                  'best score %.2f' % best_score,
                  'snake', avg_body[-1], 'steps', n_steps)

        if score > best_score:
            best_score = score

    env.close()

    if not learn:
        print('final score :\n score: %.2f' % np.mean(scores),
              'snake:%.2f' % np.mean(avg_body))

    return (np.mean(scores))
