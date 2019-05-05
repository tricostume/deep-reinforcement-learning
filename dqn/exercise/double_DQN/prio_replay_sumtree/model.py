import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl1 = nn.Linear(8, 64)
        self.fcl2 = nn.Linear(64, 128)
        self.fcl3 = nn.Linear(128, 4)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fcl1(state))
        x = F.relu(self.fcl2(x))
        x = (self.fcl3(x))
        return x