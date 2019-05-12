# Ray based applied navigation

DISCLOSURE: This is not my main repository and is used only for delivering project reports.

This constitutes a project aiming for solving the problem of applied navigation in an environment with obstacles. In this sense an agent navigates in a 2D world attempting to pick only to pick yellow bananas while rejecting to pick the blue ones. The problem is approached from reinforcement learning perspective, in particular with methods based on DQN.

For a more comprehensive explanation on the Know How refer to the <a href='REPORT.md'>Technical Report</a><br>
## The environment

The environment consists of a 2D squared world in which yellow and blue bananas lay on the floor. New bananas fall randomly at different times during the development of the interaction and it is the task of the agent to collect as many yellow bananas as possible.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an **average score of +13 over 100 consecutive episodes.**

## Installation instructions

If training is desired, it suffices to run Navigation.ipynb (first cell). The last cell loads the pretrained model and uses it for testing in the simulation environment.
The code is structured in the next way:

Navigation.ipynb: Main function. Just need to run one cell for training or another one for testing with the train model.
prio_rep_double_dqn_agent.py: Contains the implementation of the agent.
model2.py: Contains the implementation of the deep neural network with torch.
prioritized_memory.py: Contains a personalized interface to a custom memory.
SumTree.py: Contains the particular implementation of the memory used for replay.

Specific instructions are shown inside of the jupyter notebook but this information is enough to run the code.
