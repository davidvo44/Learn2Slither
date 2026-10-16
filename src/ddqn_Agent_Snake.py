import torch
from replayBufferClass import ReplayBuffer
from src.dqn import DQN
import numpy as np
import random
import os


class DDQNAgent(object):
    def __init__(self,
                 n_actions,
                 input_dims,
                 hp,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.99998,
                 batch_size=512,
                 target_update_freq=1000,
                 mem_size=100000,
                 episodes=1000,
                 replace=2000,
                 algo=None
                 ):

        self.device = torch.device('cpu')
        self.last_move = -1
        self.mem_cntr = 0
        self.learn_step_counter = 0
        self.lr = hp['lr']
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_min
        self.eps_dec = hp['epsilon_decay']
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_size = mem_size
        self.episodes = episodes
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DQN(lr=self.lr, n_action=self.n_actions,
                          input_dim=self.input_dims).to(self.device)
        self.q_next = DQN(lr=self.lr, n_action=self.n_actions,
                          input_dim=self.input_dims).to(self.device)

        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()

        self.replace_target_cnt = replace
        self.algo = algo

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        # Q-Valeurs prédites pour l'état actuel et l'action choisie
        q_pred = self.q_eval(states)[indices, actions]

        # Calcul DDQN
        with torch.no_grad():
            q_eval_next = self.q_eval(states_)
            max_actions = torch.argmax(q_eval_next, dim=1)

        # Évaluation de cette action par Target Network
            q_next_preds = self.q_next(states_)
            q_next_val = q_next_preds[indices, max_actions]

            q_next_val[dones] = 0.0

        # Cible finale stabilisée
            q_target = rewards + self.gamma * q_next_val

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10.0)
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
             self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min,
                           self.eps_dec * self.epsilon)

    def choose_action(self, observation, env, mode='TRAIN', visual=False):
        valid_actions_mask = self.get_valid_actions_mask()
        valid_indices = np.where(valid_actions_mask == 1)[0]

        # Mode Exploration
        if random.random() < self.epsilon:
            self.last_move = int(random.choice(valid_indices))
            return self.last_move

        # Mode Exploitation
        state = np.array(observation, dtype=np.float32)
        state_t = torch.tensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions = self.q_eval(state_t).squeeze(0)
        if mode == 'TRAIN':
            mask_tensor = torch.tensor(valid_actions_mask, dtype=torch.bool,
                                       device=self.device)
            actions[~mask_tensor] = float('-inf')
        if visual is True:
            print('right:%.2f' % actions[0].item(),
                  'left: %.2f' % actions[2].item(),
                  'up: %.2f' % actions[1].item(),
                  'down: %.2f' % actions[3].item())
        self.last_move = torch.argmax(actions).item()
        return self.last_move

    def get_valid_actions_mask(self):
        mask = np.ones(self.n_actions, dtype=np.int8)

        OPPOSITE_ACTIONS = {
            0: 2,
            1: 3,
            2: 0,
            3: 1
        }

        if self.last_move != -1:
            forbidden_action = OPPOSITE_ACTIONS.get(self.last_move)
            if forbidden_action is not None:
                mask[forbidden_action] = 0

        return mask

    def sample_memory(self):
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        next_states =\
            torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones

    def save_model(self, filepath="snake_dqn.pth"):
        checkpoint = {
            'q_eval_state_dict': self.q_eval.state_dict(),
            'q_next_state_dict': self.q_next.state_dict(),
            'optimizer_state_dict': self.q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath="snake_dqn.pth"):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
            self.q_next.load_state_dict(checkpoint['q_next_state_dict'])
            self.q_eval.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.learn_step_counter = checkpoint['learn_step_counter']
