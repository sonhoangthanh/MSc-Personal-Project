class RuleBasedControl:
    def __init__(self, ess1_capacity, ess2_capacity, ess1_power, ess2_power, action_space):
        self.ess1_capacity = ess1_capacity
        self.ess2_capacity = ess2_capacity
        self.ess1_power = ess1_power
        self.ess2_power = ess2_power
        self.action_space = action_space

    # Rule-based controller which performs actions based on the state
    def select_action(self, state):
        
        power_balance = state[0] + state[1] - state[2]  # solar + wind - load

        # tracking variable
        available_power = power_balance

        action = [0, 0, 0]

        # Use nominal power outputs of ESS modules
        # If batery charge/discharge is available, use battery to charge/discharge
        # If afterwards, hydrogen charge/discharge is available, use hydrogen to charge/discharge is available
        # If after all this discharge, we still have deficit, use generator energy

        # Charge if power balance is positive battery is not full
        if power_balance > 0:
            if state[3] != self.ess1_capacity:
                action[0] = -20e3  # charge to batteries
                available_power -= self.ess1_power  # subtract from tracking variable
            if available_power > self.ess2_power:
                action[1] = -20e3  # charge to hydrogen storage
            return self.action_space.index(action)

        # Discharge if power balance is negative
        if power_balance < 0:
            if state[3] != 0:
                  action[0] = 20e3  # discharge from batteries
                  available_power += self.ess1_power  # add to tracking variable
            if available_power < -self.ess2_power:
                if state[4] != 0:
                    action[1] = 20e3  # discharge from hydrogen storage
                    available_power += self.ess2_power
            if available_power < 0:  # use generator if there is sill some demanded power left 
                action[2] = 1  # use generator
            return self.action_space.index(action)