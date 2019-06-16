# Continuous Control

DISCLOSURE: This is not my main repository and is used only for delivering project reports.

This constitutes a project aiming for solving the problem of set point tracking from a double joint virtual serial robot. The end effector or the tip of the hand must follow the set point given. INteresting is thence once found not to leave the set point as long as possible. The problem is approached from the Reinforcement Learning perspective by programming a DDPG agent to accomplish the task described below.

For a more comprehensive explanation on the Know How refer to the <a href='REPORT.md'>Technical Report</a><br>
## The environment

The environment consists of a board in which one or more of such virtual robots can be collocated. Each of them must track their imposed setpoints as long as possible, as reward is rewarded for this.

A reward of +0.1 is provided for each successful timestep in which the setpoint was tracked, i.e. in which the end effector of the robot stayed within boundaries of the green set point ball. 

The state space has 33 dimensions and contains the agent's joints positions and velocities as well as other variables.  Given this information, the agent has to learn how to best select actions.  Four continuous actions, corresponding to the velocities given to the joints are then the control for this task and must have a value between the limits [-1, 1]

The task is continuous but I noticed that the environment, for means of fair comparison ends an episode every 1000 steps, and in order to solve the environment, the agent must get an **average score of +30 over 100 consecutive episodes.**

## Installation instructions

If training is desired, it suffices to run Navigation.ipynb (first cell). The last cell loads the pretrained model and uses it for testing in the simulation environment.
The code is structured in the next way:

Continuous_Control.ipynb: Main function. Just need to run one cell for training or another one for testing with the train model.
DDPGAgent.py: Contains the implementation of the agent.
models.py: COntains neural network architectures for actor and critic.

Specific instructions are shown inside of the jupyter notebook but this information is enough to run the code.
