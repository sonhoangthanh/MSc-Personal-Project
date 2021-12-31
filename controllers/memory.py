from collections import deque, namedtuple
import torch
import random
import numpy as np

class ReplayMemory(object):
    """
        Implements the memory replay buffer of fixed size for batch learning.

        Inputs: 
                (int) batch_size - size of minibatch
                (int) capacity - capacity of memory buffer
    """
    def __init__(self, batch_size, capacity=1000):
        # Initialize memoery as the deque container of max size of capacity
        # Deque allows for fast appending and pops / removes
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

        # Check if the device has available gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Function which returns the length of the memory pool/list
    def __len__(self):
        return len(self.memory)

    # Function to append the transition information to the memory
    def push(self, *args):
        self.memory.append(self.experience(*args))

    # Function to randomly sample from the memory pool the batch of size 'batch_size' to be used for optimization
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)

        # Need to stack the experience fields into tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).type(torch.int64).to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).type(torch.int64).to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states)
