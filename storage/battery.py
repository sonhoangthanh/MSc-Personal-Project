# This module contains all the Energy Storage Systems module types, such as Li ion battery, hydrogen storage etc

class simple_battery:
    def __init__(self, capacity=500e3, efficiency=0.95, power=20e3):
        self.capacity = capacity
        self.charge_efficiency = efficiency
        self.discharge_efficiency = efficiency
        self.energy_state = 200e3
        self.power = power

    def charge(self):

        if self.energy_state == self.capacity:
            return 0

        output_power = self.power
        # Amount of charge obtained from charging with input power
        charge_amount = output_power * self.charge_efficiency

        # Charging amount is constrained by battery capacity
        if self.energy_state + charge_amount > self.capacity:
            # Limit the charged amount such that the capacity is achieved
            charge_amount = self.capacity - self.energy_state
            output_power = charge_amount / self.charge_efficiency
            self.energy_state = self.capacity
        else:
            self.energy_state += charge_amount
        
        return -output_power

    def discharge(self):

        if self.energy_state == 0:
            return 0

        output_power = self.power

        discharge_amount = output_power / self.discharge_efficiency

        if self.energy_state - discharge_amount < 0:
            discharge_amount = self.energy_state
            output_power = discharge_amount * self.discharge_efficiency
            self.energy_state = 0
        else:
            self.energy_state -= discharge_amount
        
        return output_power
        