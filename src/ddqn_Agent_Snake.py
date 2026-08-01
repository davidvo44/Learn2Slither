import torch.nn as nn
import torch
import torch.optim as optim
from replayBufferClass import ReplayBuffer
from src.dqn import DQN
import numpy as np
import random
import os


class DDQNAgent(object):
    def __init__(self,
                 n_actions,
                 input_dims,
                 lr=0.0005,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.999995,
                 batch_size=128,
                 target_update_freq=1000,
                 mem_size=100000,
                 episodes=1000,
                 replace=2000,
                 algo=None
                 ):
        
        self.mem_cntr = 0
        self.learn_step_counter = 0
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_min
        self.eps_dec = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_size = mem_size
        self.episodes = episodes
        self.n_actions = n_actions
        self.input_dims = input_dims
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DQN(lr=self.lr, n_action=self.n_actions,
                          input_dim=self.input_dims)
        self.q_next = DQN(lr=self.lr, n_action=self.n_actions,
                          input_dim=self.input_dims)

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

        # Calcul DDQN avec detach complet pour éviter de corrompre les gradients futurs
        with torch.no_grad():
            q_eval_next = self.q_eval(states_)
            max_actions = torch.argmax(q_eval_next, dim=1)
        
        # 2. Évaluation de cette action par le modèle cible (Target Network)
            q_next_preds = self.q_next(states_)
            q_next_val = q_next_preds[indices, max_actions]

        # 3. Si l'etat fini, pas de future
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

    def choose_action(self, observation, env):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        
        state = np.array(observation, dtype=np.float32)
        state_t = torch.tensor(state).unsqueeze(0)

        with torch.no_grad():
            actions = self.q_eval(state_t)

        return torch.argmax(actions).item()
        
    def sample_memory(self):
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        # C'est ici qu'on applique le / 255.0 au lot entier (Batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        return states, actions, rewards, next_states, dones

    def save_model(self, filepath="snake_dqn.pth"):
        """Sauvegarde les poids des réseaux et l'état de l'optimiseur"""
        checkpoint = {
            'q_eval_state_dict': self.q_eval.state_dict(),
            'q_next_state_dict': self.q_next.state_dict(),
            'optimizer_state_dict': self.q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }
        torch.save(checkpoint, filepath)
        print(f"[*] Checkpoint sauvegardé avec succès dans {filepath}")

    def load_model(self, filepath="snake_dqn.pth"):
        """Charge un checkpoint existant"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
            self.q_next.load_state_dict(checkpoint['q_next_state_dict'])
            self.q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.learn_step_counter = checkpoint['learn_step_counter']
            print(f"[*] Checkpoint chargé depuis {filepath} (Epsilon: {self.epsilon:.4f})")
        else:
            print(f"[!] Aucun fichier trouvé à l'emplacement {filepath}")