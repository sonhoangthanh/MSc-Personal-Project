import random
import numpy as np

class RandomControl:
    """Implements the random action agent which performs random actions from the pool of input action space size"""

    def __init__(self, action_size):
        self.action_size = action_size
    
    def select_action(self, state):
        return random.choice(np.arange(0, self.action_size))
