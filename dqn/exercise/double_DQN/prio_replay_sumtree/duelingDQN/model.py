import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl1 = nn.Linear(state_size, hidden_size)
        self.fcl2 = nn.Linear(hidden_size, 2*hidden_size)
        self.fcl3 = nn.Linear(2*hidden_size, 1)

        self.fcl4 = nn.Linear(hidden_size, 2*hidden_size)
        self.fcl5 = nn.Linear(2*hidden_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fcl1(state))

        # State value stream
        x1 = F.relu(self.fcl2(x))
        x1 = self.fcl3(x1)

        # Action advantage stream
        x2 = F.relu(self.fcl4(x))
        x2 = self.fcl5(x2)

        # Break intractability
        A = x2 - (torch.mean(x2) * torch.ones(self.action_size))
        V = x1 * torch.ones(self.action_size)
        return V + A