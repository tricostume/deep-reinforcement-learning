# Double DQN

The same Q network as used for vanilla DQN was used here. The proposed method by van Hasselt, Guez and Silver as an addition to the local/target network separation was implemented here. Namely the maximizing actions being chosen from the local network and their values from the target network. In this manner, as explained by the respective paper, the overestimation of the action value functions is diminished.
