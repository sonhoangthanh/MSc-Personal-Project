import pytest

from storage import battery, hydrogen, simple_battery, simple_hydrogen
from utils import load_data

@pytest.fixture
def ess1():
    return simple_battery()

@pytest.fixture
def ess2():
    return simple_hydrogen()

def test_ESS_init(ess1, ess2):

    # Test default initialization
    # Battery
    assert ess1.capacity == 500e3
    assert ess1.charge_efficiency == 0.95
    assert ess1.discharge_efficiency == 0.95
    assert ess1.energy_state == 200e3

    # Hydrogen
    assert ess2.capacity == 500e3
    assert ess2.charge_efficiency == 0.7
    assert ess2.discharge_efficiency == 0.7
    assert ess2.energy_state == 200e3


def test_charging(ess1, ess2):

    # Test normal charging
    power = 20e3  # 10 MW

    input_charge_ess1 = ess1.energy_state
    input_charge_ess2 = ess2.energy_state

    ess1_power = ess1.charge()
    ess2_power = ess2.charge()

    assert ess1.energy_state == input_charge_ess1 + ess1.charge_efficiency * power
    assert  ess2.energy_state == input_charge_ess2 + ess2.charge_efficiency * power
    assert ess1_power == -power
    assert ess2_power ==  -power

    # Test that we cant charge more than maximum deliverable power

    input_charge_ess1 = ess1.energy_state
    input_charge_ess2 = ess2.energy_state

    ess1_power = ess1.charge()
    ess2_power = ess2.charge()

    assert ess1.energy_state == input_charge_ess1 + ess1.charge_efficiency * power
    assert  ess2.energy_state == input_charge_ess2 + ess2.charge_efficiency * power
    assert ess1_power == -power
    assert ess2_power ==  -power

    # Check that charging is capped by the capacity
    ess1.energy_state = ess1.capacity - 10e3
    ess2.energy_state = ess2.capacity - 10e3
    power_ess1 = ess1.charge()
    power_ess2 = ess2.charge()

    # Charge stays the same at maximum and doesnt exceeds it
    assert ess1.energy_state == ess1.capacity
    assert ess2.energy_state == ess2.capacity
    assert power_ess1 == -10e3 / ess1.charge_efficiency
    assert power_ess2 == -10e3 / ess2.charge_efficiency 

    # Check that no more charging is allowed
    power = 10e3

    power_ess1 = ess1.charge()
    power_ess2 = ess2.charge()

    assert ess1.energy_state == ess1.capacity
    assert ess2.energy_state == ess2.capacity
    assert power_ess1 == 0
    assert power_ess2 == 0
    

def test_discharge(ess1, ess2):

    # Set the charge states to the maximum
    ess1.energy_state = ess1.capacity
    ess2.energy_state = ess2.capacity

    # Check that discharging zero power results in zero
    result1 = ess1.discharge()
    result2 = ess2.discharge()
    assert result1 == 20e3
    assert result2 == 20e3

    # Check that you can't discharge if you have zero charge
    ess1.energy_state = 0
    ess2.energy_state = 0
    
    result1 = ess1.discharge()
    result2 = ess2.discharge()
    assert result1 == 0
    assert result2 == 0

    ess1.energy_state = 10e3
    ess2.energy_state = 10e3

    result1 = ess1.discharge()
    result2 = ess2.discharge()

    assert ess1.energy_state == 0
    assert ess2.energy_state == 0

    assert result1 == 10e3 * ess1.discharge_efficiency
    assert result2 == 10e3 * ess2.discharge_efficiency

