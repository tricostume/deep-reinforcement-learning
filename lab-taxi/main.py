from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')
agent = Agent(nA=6,\
              strategy="expected_sarsa",\
              GLIE="dec_bias",\
              GLIE_param=0.001,\
              alpha = 0.6,\
              gamma = 1)

num_episodes = 20000
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes)

plt.plot(np.linspace(0,num_episodes,len(avg_rewards),endpoint=False), np.asarray(avg_rewards))
plt.xlabel('Episode Number')
plt.ylabel('Episode Reward')
plt.show()
