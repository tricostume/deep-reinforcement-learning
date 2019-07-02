import numpy as np
import random
from collections import namedtuple, deque
from models import Actor, Critic
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, a_check=None, c_check=None, gamma=0.99, tau=1e-3, add_noise=False, mu=0.,
                 theta=0.15, sigma=0.1, lr_actor=2e-4, lr_critic=2e-4,buffer_size=1e5, batch_size=128, update_every = 1,
                 low_action=-1, high_action=1, num_agents=1, warm_up=0, consecutive_learns=1):
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
        self.random_process = OUNoise(action_size, seed, mu=mu, theta=theta, sigma=sigma)
        self.gamma = gamma
        self.tau = tau
        # Actor and Critic approximators
        self.targetActor = Actor(state_size, action_size, seed, (400,300)).to(device)
        self.targetCritic = Critic(2*state_size, 2*action_size, seed, (400,300)).to(device)
        self.actor = Actor(state_size, action_size, seed, (400, 300)).to(device)
        self.critic = Critic(2*state_size, 2*action_size, seed, (400, 300)).to(device)
        for target, local in zip(self.targetCritic.parameters(), self.critic.parameters()):
            target.data.copy_(local.data)
        for target, local in zip(self.targetActor.parameters(), self.actor.parameters()):
            target.data.copy_(local.data)

        if a_check is not None:
            self.actor.load_state_dict(a_check)
        if c_check is not None:
            self.critic.load_state_dict(c_check)
            
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t = 0
        self.warm_up = warm_up
        self.add_noise = add_noise
        self.update_every = update_every
        self.low_action = low_action
        self.high_action = high_action
        self.num_agents = num_agents
        self.consecutive_learns = consecutive_learns

    def reset(self):
        self.random_process.reset()

    def act(self, state, random=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            eval (boolean) : Turns off mean and std deviation from evaluation batches if set to true
        """
        if random is True or self.t < self.warm_up:
            action = np.random.randn(self.num_agents, self.action_size)
        else:
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
                if self.add_noise:
                    noise = self.random_process.sample()
                    action += noise
            self.actor.train()
        return np.clip(action, self.low_action, self.high_action)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory NUMPY
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t > self.warm_up:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % self.update_every
            if len(self.memory) > self.memory.batch_size:
                for i in range(0,self.consecutive_learns):
                    # If enough samples are available in memory, get random subset and learn
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.targetActor(next_states)
        Q_targets_next = self.targetCritic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss (using gradient clipping)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.targetCritic, self.tau)
        self.soft_update(self.actor, self.targetActor, self.tau)       
        
        '''
        print(next_action_values_local.shape)
        print(next_action_values_local[0][:])
        print(next_action_values_local.gather(1, actions).shape)
        print(actions[0][0])
        print(next_action_values_local.gather(1, actions)[0][0])
        '''
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def adjust_learning_rate(self, episode, val):
        print("adjusting learning rate!")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = val

#----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


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
        self.memory = deque(maxlen=int(buffer_size))
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state