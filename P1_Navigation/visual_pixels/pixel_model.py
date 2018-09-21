import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, num_input_chnl, action_size, seed, num_filters = [16,32], fc_layers=[64,64]):
        """Initialize parameters and build model.
        Params
        ======
            num_input_chnl (int): Number of input channels
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(num_input_chnl, num_filters[0], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv1bnorm = nn.BatchNorm2d(num_filters[0])
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #self.conv2d_1 = [self.conv1, self.bnorm1, self.relu1, self.maxp1]

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv2bnorm = nn.BatchNorm2d(num_filters[1])
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(num_filters[1]*21*21, fc_layers[0])
        self.fc1bnorm = nn.BatchNorm1d(fc_layers[0])
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc2bnorm = nn.BatchNorm1d(fc_layers[1])
        self.fc2relu = nn.ReLU()
        
        self.fc3 = nn.Linear(fc_layers[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # for conv_1 in self.conv2d_1:
        #     state = conv_1(state)

        state = self.conv1(state)
        state = self.conv1bnorm(state)
        state = self.conv1relu(state)
        state = self.conv1maxp(state)

        state = self.conv2(state)
        state = self.conv2bnorm(state)
        state = self.conv2relu(state)
        state = self.conv2maxp(state)

        #print(state.shape) #state is of shape Nx32x21x21
        state = state.reshape((-1,32*21*21)) #reshape the output of conv2 before feeding into fc1 layer

        state = self.fc1(state)
        state = self.fc1bnorm(state)
        state = self.fc1relu(state)

        state = self.fc2(state)
        state = self.fc2bnorm(state)
        state = self.fc2relu(state)

        state = self.fc3(state)

        return state

'''
Note: when training, do model_name.train() to properly update batchnorm variables. 
And during inference, do model_name.eval() to us the batch norm statistics from training time.
The dqn_agent's act method already handles this.

To speed up inference turn off gradients like this:
with torch.no_grad():
    action = model.forward(state)

'''

# If it doesn't work, maybe remove batchnorm.

