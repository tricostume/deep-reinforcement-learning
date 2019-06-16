import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=(400,300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl1 = nn.Linear(state_size, hidden_sizes[0])
        self.fcl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fcl3 = nn.Linear(hidden_sizes[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcl1.weight.data.uniform_(*hidden_init(self.fcl1))
        self.fcl2.weight.data.uniform_(*hidden_init(self.fcl2))
        self.fcl3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        x = F.relu(self.fcl1(state))
        x = self.bn1(x)
        x = F.relu(self.fcl2(x))
        return F.tanh(self.fcl3(x))


class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=(512,256,128)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl1 = nn.Linear(state_size, hidden_sizes[0])
        self.fcl2 = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.fcl3 = nn.Linear(hidden_sizes[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcl1.weight.data.uniform_(*hidden_init(self.fcl1))
        self.fcl2.weight.data.uniform_(*hidden_init(self.fcl2))
        self.fcl3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fcl1(state))
        x = self.bn1(x) 
        x = F.relu(self.fcl2(torch.cat([x, action], dim=1)))
        return self.fcl3(x)