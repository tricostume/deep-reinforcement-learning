import numpy as np
import random
from collections import namedtuple, deque
from models import Actor, Critic
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

UPDATE_EVERY = 1  # how often to update the network
LOW_ACTION = -1
HIGH_ACTION = 1
NUM_AGENTS = 1
WARM_UP = 0
CONSECUTIVE_LEARNS = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, a_check=None, c_check=None, gamma=0.99, tau=1e-3, add_noise=False, mu=0.,
                 theta=0.15, sigma=0.1, lr_actor=2e-4, lr_critic=2e-4,buffer_size=1e5, batch_size=128 ):
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
        self.targetActor = Actor(state_size, action_size, seed, (128,128)).to(device)
        self.targetCritic = Critic(state_size, action_size, seed, (128,128)).to(device)
        self.actor = Actor(state_size, action_size, seed, (128, 128)).to(device)
        self.critic = Critic(state_size, action_size, seed, (128, 128)).to(device)
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
        self.warm_up = WARM_UP
        self.add_noise = add_noise

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
            action = np.random.randn(NUM_AGENTS, self.action_size)
        else:
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
                if self.add_noise:
                    noise = self.random_process.sample()
                    action += noise
            self.actor.train()
        return np.clip(action, LOW_ACTION, HIGH_ACTION)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory NUMPY
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t > self.warm_up:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if len(self.memory) > self.memory.batch_size:
                for i in range(0,CONSECUTIVE_LEARNS):
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


class MeanStdNormalizer:
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

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


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val