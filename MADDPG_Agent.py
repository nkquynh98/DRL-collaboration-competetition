import numpy as np
import random
import copy
from collections import namedtuple, deque

from MADDPG_Model import MA_Actor, MA_Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 4e-4         # learning rate of the actor 
LR_CRITIC = 8e-4        # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay 0.001
UPDATE_EVERY = 1       # Update after t-step
NUM_UPDATE = 1          # Number of updates per step  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.num_agents = num_agents
        self.time_step = 0 #Counter for update every UPDATE_EVERY step
        # Actor Network (w/ Target Network)
        # Create the network for each agent
        self.actor_local = [MA_Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_target = [MA_Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_optimizer = [optim.Adam(self.actor_local[i].parameters(), lr=LR_ACTOR) for i in range(num_agents)]

        # Critic Network (w/ Target Network)
        # Create Network for each critic, those critic networks contains joint_state, joint_action
        self.critic_local = [MA_Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critic_target = [MA_Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critic_optimizer = [optim.Adam(self.critic_local[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for i in range(num_agents)]

        # Noise process
        #self.noise = OUNoise((self.num_agents,action_size), random_seed)
        self.noise = OUNoise((self.num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(num_agents*action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step+1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and (self.time_step == 0):
            for i in range(NUM_UPDATE):
                #experiences = self.memory.sample()
                self.learn(GAMMA)

    def act(self, states, eps, add_noise=True):

        """Returns actions for given multiple states as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        for actor_local in self.actor_local:
            actor_local.eval()
        with torch.no_grad():
            actions = np.array([self.actor_local[i](states[i].unsqueeze(0)).cpu().data.numpy().squeeze() for i in range(self.num_agents)]) 
        for actor_local in self.actor_local:    
            actor_local.train()
        if add_noise:
            actions += eps * self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        for i in range(self.num_agents):
            #Sample a batch of experience for each agent
            states, actions, rewards, next_states, dones = self.memory.sample()

            # ---------------------------- update critic ---------------------------- #

            # Create the joint_state, joint_action, joint_nextstate for updating the common critic
            # In other words, the dimension is reduce as follow: batch x agents x  state --> batch x (agent*state)

            joint_states = states.reshape(states.shape[0], -1) 
            joint_actions = actions.reshape(actions.shape[0], -1)
            joint_next_states = next_states.reshape(next_states.shape[0], -1)

            rewards_local = rewards[:, i, :]
            dones_local = dones[:, i, :]
            # Get predicted next-state actions and Q values from target models
            with torch.no_grad():
                actions_next = torch.stack([self.actor_target[j](states[:, j, :]) for j in range(self.num_agents)], dim = 1) 
                joint_actions_next = actions_next.reshape(actions_next.shape[0], -1)
                Q_targets_next = self.critic_target[i](joint_next_states, joint_actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards_local + (gamma * Q_targets_next * (1 - dones_local))
            # Compute critic loss
            # print(rewards_local.shape)
            # print(dones_local.shape)
            # print(Q_targets_next.shape)
            # print(Q_targets.shape)
            Q_expected = self.critic_local[i](joint_states, joint_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local[i].parameters(), 1)
            self.critic_optimizer[i].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss

            actions_pred = []
            #np.array([self.actor_local[j](states[:, j, :])])
            for j in range(self.num_agents):
                if j == i:
                    actions_pred.append(self.actor_local[j](states[:, j, :]).contiguous())
                else:
                    actions_pred.append(actions[:, j, :])
            actions_pred = torch.stack(actions_pred, dim = 1)
            #print(actions_pred.shape)
            joint_actions_pred = actions_pred.reshape(actions_pred.shape[0], -1)
            #actions_pred = np.array([self.actor_local[j](states[:, j, :]) 
            actor_loss = -self.critic_local[i](joint_states, joint_actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()

            # ----------------------- update target networks ----------------------- #
            
            self.soft_update(self.critic_local[i], self.critic_target[i], TAU)
            self.soft_update(self.actor_local[i], self.actor_target[i], TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0., sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        #return self.state
        return dx

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis = 0)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis = 0)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis = 0)).float().unsqueeze(2).to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis = 0)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis = 0).astype(np.uint8)).float().unsqueeze(2).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)