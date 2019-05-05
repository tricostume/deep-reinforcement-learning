import numpy as np
import random
from collections import namedtuple, deque
from prio_rep_mem import SuperTuple, MySortedList
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
REPLAY_ALPHA = 0.6      # alpha used for prioritzed replay
REPLAY_BETA = 0.0       # beta used for prioritized replay
REPLAY_BETA_INCREMENT = 0.0005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, idxs, ws, probs = self.memory.sample()
                self.learn(experiences, idxs, ws, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxs, ws, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        next_action_values_local = self.qnetwork_local(states).gather(1, actions)
        # Only change proposed for Double DQN: Get maximizing future actions from local network and get their
        # corresponding values from target network. Compare then these to the local taken actions.
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        next_action_values_target = self.qnetwork_target(next_states).detach().gather(1, local_max_actions)
        
        
        '''
        print(next_action_values_local.shape)
        print(next_action_values_local[0][:])
        print(next_action_values_local.gather(1, actions).shape)
        print(actions[0][0])
        print(next_action_values_local.gather(1, actions)[0][0])
        '''
        y = rewards + (gamma * next_action_values_target*(1 - dones))
        # Local network will be actualized, target one is used as ground truth
        ws = torch.from_numpy(ws.astype(float)).float().to(device)
        loss = F.mse_loss(ws*next_action_values_local, ws*y)
        errors = np.abs(y.cpu().data.numpy() - next_action_values_local.cpu().data.numpy())
        self.memory.memory.update_batch(idxs, errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        # Copy from local to target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = MySortedList(buf_len=buffer_size, bins_num=batch_size, replay_alpha=REPLAY_ALPHA, replay_beta = REPLAY_BETA, rep_beta_inc=REPLAY_BETA_INCREMENT)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if len(self.memory) <= self.batch_size:
            error = random.random()
        else:
            error = self.memory.sl[0].d['p']
        e = SuperTuple(state, action, reward, next_state, done, error, error**REPLAY_ALPHA)
        self.memory.add(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences, idxs, ws, probs = self.memory.sample(bins_num=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.d['t'].state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.d['t'].action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.d['t'].reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.d['t'].next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.d['t'].done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones), idxs, ws, probs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)