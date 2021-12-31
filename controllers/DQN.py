import torch.nn as nn
import torch.nn.functional as F
import torch


# The fully connected neural network to be used for finding policies and performing actions
class DQN(nn.Module):
    """ 
        Deep Q Neural Network. 
        The inputs are:
        - PV power generated
        - Wind power generated
        - Load
        - State of charge in ESS1
        - State of Charge in ESS2

        The outputs are 18 actions consisting of:
        - Charge ESS1 / Discharge ESS1
        - Charge ESS2 / Discharge ESS2
        - Use Power Generator
    """
    # Init function
    def __init__(self, observation_space, action_space, l1_size=18, l2_size=90, l3_size=18, l4_size=50):
        # Supercharge the function
        super(DQN, self).__init__()
        self.l1 = nn.Linear(observation_space, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, l3_size)
        self.l4 = nn.Linear(l3_size, l4_size)
        self.activate = nn.ReLU()
        self.output = nn.Linear(l2_size, action_space)

    # Forward function
    def forward(self, x):
        # Put the value to device
        x1 = self.activate(self.l1(x))
        x2 = self.activate(self.l2(x1))
        # x3 = self.activate(self.l3(x2))
        # x4 = self.activate(self.l4(x3))
        return self.output(x2)
