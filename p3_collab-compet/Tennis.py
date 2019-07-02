from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import torch
from collections import deque
from maddpg import MADDPG

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
'''
count = 0
import time
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        count += 1
        #time.sleep(3)
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        #print("Agent 1:")
        #print(str(next_states[0]))
        #print("Agent 2:")
        #print(str(next_states[1]))
        #print()
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones) or count==10:                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

env.close()
'''

# Create an agent, pass a desired size for the hiden layers.
agent = MADDPG(state_size, action_size, seed=10, a_check=None, c_check=None, gamma=0.995, tau=1e-3, add_noise=False, mu=0.,
                 theta=0.15, sigma=0.2, lr_actor=1e-4, lr_critic=4.2e-3,buffer_size=1e5, batch_size=200, update_every = 4,
                 low_action=-1, high_action=1, num_agents=2, warm_up=0, consecutive_learns=3, clip_critic_grad=0)


# Define dqn algorithm
def maddpg(n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(2)
        while True:
            actions = agent.act(states, random=False)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done

            #next_state = agent.state_normalizer(next_state)
            #reward = agent.reward_normalizer(reward)

            agent.step(states, actions, rewards, next_states, dones, i_episode)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        episode_score = np.max(score)
        scores_window.append(episode_score)  # save most recent score
        scores.append(episode_score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tlast score: {:.2f}'.format(i_episode, np.mean(scores_window), episode_score), end="")
        if i_episode % 100 == 0:
            print('\nEpisode {}\tAverage Score: {:.2f}\tlast score: {:.2f}'.format(i_episode, np.mean(scores_window), episode_score), end="")
        if np.mean(scores_window) >= 0.5 and i_episode > 50:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.critic.state_dict(), 'my_critic.pth')
            torch.save(agent.actor.state_dict(), 'my_actor.pth')
            break
        # A small step in learning rate to allow for quicker convergence with above set parameters
        #if i_episode == 1200:
        #    agent.adjust_learning_rate(1200, 2E-5)
    return scores


scores = maddpg()

env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Save scores
with open('scores.txt', 'w') as f:
    for item in scores:
        f.write("%f\n" % item)


'''
agent = MADDPG(state_size, action_size, seed=10, a_check=None, c_check=None, gamma=0.995, tau=1e-3, add_noise=False, mu=0.,
                 theta=0.15, sigma=0.2, lr_actor=1e-4, lr_critic=4e-3,buffer_size=1e5, batch_size=200, update_every = 4,
                 low_action=-1, high_action=1, num_agents=2, warm_up=0, consecutive_learns=3, clip_critic_grad=0)
                 
Episode 100	Average Score: 0.03	last score: 0.00
Episode 200	Average Score: 0.00	last score: 0.00
Episode 300	Average Score: 0.02	last score: 0.00
Episode 400	Average Score: 0.01	last score: 0.00
Episode 500	Average Score: 0.01	last score: 0.00
Episode 600	Average Score: 0.00	last score: 0.00
Episode 700	Average Score: 0.01	last score: 0.00
Episode 800	Average Score: 0.04	last score: 0.00
Episode 900	Average Score: 0.01	last score: 0.00
Episode 1000	Average Score: 0.07	last score: 0.10
Episode 1100	Average Score: 0.09	last score: 0.09
Episode 1200	Average Score: 0.08	last score: 0.10
Episode 1300	Average Score: 0.08	last score: 0.10
Episode 1400	Average Score: 0.11	last score: 0.10
Episode 1500	Average Score: 0.10	last score: 0.10
Episode 1600	Average Score: 0.12	last score: 0.10
Episode 1700	Average Score: 0.08	last score: 0.00
Episode 1800	Average Score: 0.08	last score: 0.10
Episode 1900	Average Score: 0.11	last score: 0.20
Episode 2000	Average Score: 0.08	last score: 0.10
Episode 2100	Average Score: 0.23	last score: 0.10
Episode 2200	Average Score: 0.14	last score: 0.10
Episode 2300	Average Score: 0.10	last score: 0.10
Episode 2400	Average Score: 0.24	last score: 0.20
Episode 2500	Average Score: 0.31	last score: 0.10
Episode 2600	Average Score: 0.29	last score: 1.00
Episode 2700	Average Score: 0.54	last score: 0.00
Episode 2703	Average Score: 0.51	last score: 0.00
'''