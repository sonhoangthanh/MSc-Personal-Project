import pytest
from environment import HRES
from storage import simple_battery, simple_hydrogen, generator

import numpy as np
import copy


# Tests for checking the correct setup of the environment

def test_HRES_value():
    """ Tests some of the 18 possible actions on the input initial state of the training environment.""" 


    data_path = './data/'
    env = HRES(data_path, mode='train', reward='revenue')  # test using value function
    
    # Test the init
    # Energy storage system
    assert type(env.ess1) == type(simple_battery())
    assert type(env.ess2) == type(simple_hydrogen())

    # Backup generator
    assert type(env.generator) == type(generator())
    
    assert env.ess1.energy_state == 200e3
    assert env.ess2.energy_state == 200e3
    
    # Time counter
    assert env.time == 0

    assert env.state == None


    # Test the step function
    # We know for the first value, there is a surplus of energy from the graph
    env.reset()
    action = env.action_space.index([0, 0, 0]) # equivalent to action = [0, 0, 0]
    env.perform_action(action)
    assert env.ess1.energy_state == 200e3
    assert env.ess2.energy_state == 200e3


    # reset the environment
    env.reset()
    state = env.observe()
    assert len(state) == 5

    action = env.action_space.index([0, 0, 1])  # equivalent to [0, 0, 1] 

    # Charge remains unchanged
    assert env.ess1.energy_state == 200e3
    assert env.ess2.energy_state == 200e3

    env.reset()
    reward, next_state = env.step(action)

    assert reward < 0
    assert len(next_state) == 5



