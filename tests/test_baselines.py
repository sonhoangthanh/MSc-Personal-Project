from random import random
import pytest
import numpy as np
from controllers import RandomControl, RuleBasedControl
from environment import HRES


def test_random_control():
    state = [0, 0, 0, 0, 0]
    agent = RandomControl(18)
    action = agent.select_action(state)
    assert action in np.arange(0, 18)


def test_rule_control():
    battery_capacity = 200e3
    battery_power = 20e3
    hydrogen_capacity = 300e3
    hydrogen_power = 20e3
    
    data_path = './data/'
    env = HRES(data_path, mode='train', reward='revenue')

    rule_controller = RuleBasedControl(battery_capacity, hydrogen_capacity, battery_power, hydrogen_power, env.action_space)
    
    # Charging ess1 only
    solar_power = 10e3
    wind_power = 40e3
    load = 40e3
    battery_energy = 100e3
    hydrogen_energy = 100e3

    state = [solar_power, wind_power, load, battery_energy, hydrogen_energy]
    action_space = [[ess1, ess2, gen] for ess1 in [0, 1, 2] for ess2 in [0, 1, 2] for gen in range(2)]

    rule_action = rule_controller.select_action(state)

    assert rule_action in np.arange(0, 18)