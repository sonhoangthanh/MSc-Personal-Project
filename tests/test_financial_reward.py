import pytest
import numpy as np
from rewards import FinancialModel

def test_financial_reward():

    financial_model = FinancialModel()

    action_space = [[ess1, ess2, gen] for ess1 in [0, 1, 2] for ess2 in [0, 1, 2] for gen in range(2)]

    # Total supplied power, including charge/discharge of ESS and power_generator
    total_supply_power = 36e3
    load = 33e3
    action = [1, 0, 0]
    generator_power = 30e3

    reward, power_loss = financial_model.get_reward(total_supply_power, load, action, 0)

    # Expected reward = consumer revenue - deviation costs - maintainance costs - generator costs - power loss penalty
    expected_reward = load * financial_model.electricity_price - abs(total_supply_power - load) * financial_model.surplus_cost - financial_model.ess_cost
    assert reward == expected_reward

    # Change the action and check the change in expected reward
    action = [1, 1, 1]
    reward, power_loss = financial_model.get_reward(total_supply_power, load, action, generator_power)

    expected_reward = load * financial_model.electricity_price - abs(total_supply_power - load) * financial_model.surplus_cost \
                                - 2 * financial_model.ess_cost - financial_model.generator_cost * generator_power

    assert reward == expected_reward

    # Check if supplied power doesnt meet the load
    total_supply_power = 30e3
    load = 33e3
    action = [0, 0, 0]

    reward, power_loss = financial_model.get_reward(total_supply_power, load, action, 0)
    expected_reward = total_supply_power * financial_model.electricity_price - abs(total_supply_power - load) * financial_model.deficit_cost - financial_model.power_loss_penalty
    assert reward == expected_reward


