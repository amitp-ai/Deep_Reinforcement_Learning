import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64,64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        #"*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layers[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.fc1(state)
        state = self.relu1(state)
        state = self.fc2(state)
        state = self.relu2(state)
        state = self.fc3(state)
        return state

