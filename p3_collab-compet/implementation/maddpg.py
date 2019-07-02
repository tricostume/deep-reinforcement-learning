import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
from utils import encode, decode
import torch.nn.functional as F
from DDPGAgent import DDPGAgent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    """MADDPG Agent : Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, a_check=None, c_check=None, gamma=0.99, tau=1e-3, add_noise=False, mu=0.,
                 theta=0.15, sigma=0.1, lr_actor=2e-4, lr_critic=2e-4,buffer_size=1e5, batch_size=128, update_every = 1,
                 low_action=-1, high_action=1, num_agents=2, warm_up=0, consecutive_learns=1, clip_critic_grad=0):
        """Initialize a MADDPG Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """

        super(MADDPG, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        # Instantiate Multiple  Agent
        self.agents = [DDPGAgent(state_size, action_size, seed, a_check=a_check[i], c_check=c_check[i], gamma=gamma, tau=tau, add_noise=add_noise, mu=mu,
                                 theta=theta, sigma=sigma, lr_actor=lr_actor, lr_critic=lr_critic,buffer_size=buffer_size, batch_size=batch_size, update_every = update_every,
                                 low_action=low_action, high_action=high_action, num_agents=num_agents, warm_up=warm_up, consecutive_learns=consecutive_learns)
                       for i in range(num_agents)]

        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # PARAMETERS
        self.batch_size = batch_size
        self.consecutive_learns = consecutive_learns
        self.gamma = gamma
        self.clip_critic_grad = clip_critic_grad
        self.update_every = update_every
        self.tau = tau

    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states, random=False):
        """Return action to perform for each agents (per policy)"""
        return [agent.act(state, random=random) for agent, state in zip(self.agents, states)]

    def step(self, states, actions, rewards, next_states, dones, num_current_episode):
        """ # Save experience in replay memory, and use random sample from buffer to learn"""

        # self.memory.add(states, It mainly reuse function from ``actions, rewards, next_states, dones)
        self.memory.add(encode(states),
                        encode(actions),
                        rewards,
                        encode(next_states),
                        dones)

        # If enough samples in the replay memory and if it is time to update
        if (len(self.memory) > self.batch_size) and (num_current_episode % self.update_every == 0):

            # Note: this code only expects 2 agents
            assert (len(self.agents) == 2)

            # Allow to learn several time in a row in the same episode
            for i in range(self.consecutive_learns):
                # Sample a batch of experience from the replay buffer
                experiences = self.memory.sample()
                # Update Agent #0
                self.maddpg_learn(experiences, own_idx=0, other_idx=1, gamma=self.gamma)
                # Sample another batch of experience from the replay buffer
                experiences = self.memory.sample()
                # Update Agent #1
                self.maddpg_learn(experiences, own_idx=1, other_idx=0, gamma=self.gamma)

    def maddpg_learn(self, experiences, own_idx, other_idx, gamma):
        """
        Update the policy of the MADDPG "own" agent. The actors have only access to agent own
        information, whereas the critics have access to all agents information.

        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            own_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # Filter out the agent OWN states, actions and next_states batch
        own_states = decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states)

        # Filter out the OTHER agent states, actions and next_states batch
        other_states = decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)

        # Concatenate both agent information (own agent first, other agent in second position)
        all_states = torch.cat((own_states, other_states), dim=1).to(device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(device)

        agent = self.agents[own_idx]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        all_next_actions = torch.cat((agent.targetActor(own_states), agent.targetActor(other_states)),
                                     dim=1).to(device)
        Q_targets_next = agent.targetCritic(all_next_states, all_next_actions)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = agent.critic(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_opt.zero_grad()
        critic_loss.backward()
        if self.clip_critic_grad:
            torch.nn.utils.clip_grad_norm(agent.critic.parameters(), 1)
        agent.critic_opt.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor(own_states), agent.actor(other_states).detach()),
                                     dim=1).to(device)
        actor_loss = -agent.critic(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic, agent.targetCritic, self.tau)
        agent.soft_update(agent.actor, agent.targetActor, self.tau)
'''
    def checkpoints(self):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'model_dir/checkpoint_actor_local_' + str(idx) + '.pth'
            critic_local_filename = 'model_dir/checkpoint_critic_local_' + str(idx) + '.pth'
            actor_target_filename = 'model_dir/checkpoint_actor_target_' + str(idx) + '.pth'
            critic_target_filename = 'model_dir/checkpoint_critic_target_' + str(idx) + '.pth'
            torch.save(agent.actor_local.state_dict(), actor_local_filename)
            torch.save(agent.critic_local.state_dict(), critic_local_filename)
            torch.save(agent.actor_target.state_dict(), actor_target_filename)
            torch.save(agent.critic_target.state_dict(), critic_target_filename)
'''

