# Collaboration and Competition

DISCLOSURE: This is not my main repository and is used only for delivering project reports.

We aim to approach the problem of dealing with sparse reward when training agents with Reinforcement Learning methods. In this application, a game of tennins in a multiagent scenario (two players) is considered. As seen afterwards given the characteristics of the environment, the behavior of even plazing is enforced i.e. to trz to maximize the time the ball is on game. This solution approaches the problem with a MADDPG (Multi Agent Deep Determinist Policy Gradients) architecture and is explained in the respective report.

For a more comprehensive explanation on the Know How refer to the <a href='REPORT.md'>Technical Report</a><br>
## The environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Installation instructions

Make sure you installed gym repository and the necessary dependencies for Udacity repository. This will get you in the right track.
If training is desired, it suffices to run Tennis_solution.ipynb (first cell). The last cell loads the pretrained model and uses it for testing in the simulation environment.
The code is structured in the next way:

Tennis_solution.ipynb: Main function. Just need to run one cell for training or another one for testing with the train model.
DDPGAgent.py: Contains a generic DDPG agent and the specific implementations of the randomic buffer replay and the noise source.
maddpg.py: Encompasses the listing of several agents to work in the multi agent sense. For this interfacing functions are created to work with stacked observations, actions, rewards and replay members. In the same manner the MADDPG algorithm for training and inference is programmed by storing the deep neural networks in a convenient manner.
models.py: Contains the proposed models for the actor and the critic, characteristic of a DDPG agent.
utils.py: Contains some helper functions for encoding and decoding the replay members to allow for stacked data structures to be used as entities.


Specific instructions are shown inside of the jupyter notebook but this information is enough to run the code.
