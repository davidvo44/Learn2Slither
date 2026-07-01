from dqn_class import DQN
import gymnasium as gym
import torch.optim as optim
import torch
from hyperparameter_class import HyperParameters
import matplotlib.pyplot as plt
import random
from collections import deque
import torch.nn as nn
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np


def dqnCartpole(hyperparam: HyperParameters):
    env = gym.make("CartPole-v1",
                   render_mode="rgb_array")

    training_period = 100

    env = RecordVideo(
        env,
        video_folder='CartPole-agent',
        name_prefix='train',
        episode_trigger=lambda x: x % training_period == 0
    )

    env = RecordEpisodeStatistics(env)
    # Initialize Q-networks
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_net = DQN(input_dim, output_dim, hyperparam.lr)
    target_net = DQN(input_dim, output_dim, hyperparam.lr)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    rewards_per_episode = []
    steps_done = 0

    memory = deque(maxlen=hyperparam.memory_size)

    for episode in range(hyperparam.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        if episode % 100 == 0:
            print(episode)
        while not done:
            action = select_action(state, hyperparam.epsilon, env, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            optimize_model(memory, hyperparam, policy_net, target_net)

            if steps_done % hyperparam.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            steps_done += 1

        hyperparam.epsilon = max(hyperparam.epsilon_min,
                                 hyperparam.epsilon_decay *
                                 hyperparam.epsilon)

        rewards_per_episode.append(episode_reward)

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole')
    plt.savefig('dqncartpole.png')


# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon, env, policy_net):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()  # Exploit


# Function to optimize the model using experience replay
def optimize_model(memory, hyperparam: HyperParameters, policy_net, target_net):
    if len(memory) < hyperparam.batch_size:
        return
    batch = random.sample(memory, hyperparam.batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + hyperparam.gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    policy_net.optimizer.zero_grad()
    loss.backward()
    policy_net.optimizer.step()
