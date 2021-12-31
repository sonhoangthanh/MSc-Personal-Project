class FinancialModel:
    """ Financial model for reward function

    Returns: Operational revenue (revenue minus costs, maintainance and penalties) from operating a HRES system, the boolian value indicating power loss

    Consumer electricity price taken from https://www.manxutilities.im. Price and costs are in british pounds per kWh:
    Electricity price = 16.9 p for kWh or 0.169£
    Generator cost = 20p for kWh or 0.2£
    Power deficit cost = 20p per kWh or 0.1£
    Power surplus cost = 20p per kWh or 0.1£
    Power loss penalty (fixed) = 1500£
    ESS maintainance cost (fixed) = 100£ for 1h of usage

    """

    def __init__(self, electricity_price=0.169, generator_cost=0.2, ess_cost=100, deficit_cost=0.2, surplus_cost=0.1, power_loss_penalty=1000):
        self.electricity_price = electricity_price  # price per kWh
        self.generator_cost = generator_cost
        self.ess_cost = ess_cost
        self.surplus_cost = surplus_cost
        self.deficit_cost = deficit_cost
        self.power_loss_penalty = power_loss_penalty

    def get_reward(self, total_power, load, action, generator_power):
        
        revenue = 0
        demanded_power = 0
        power_loss = False
        power_difference = total_power - load

        # Calculate how much power was demanded, limit is given by the load
        if total_power - load >= 0:
            demanded_power = load
        else:  # limit to load amount if power supply exceeds it
            demanded_power = total_power

        # Demanded power revenue
        revenue += self.electricity_price * demanded_power

        # Deficit cost
        if power_difference < 0:
            revenue -= self.deficit_cost * abs(power_difference)

        # Surplus cost
        if power_difference > 0:
            revenue -= self.surplus_cost * abs(power_difference)
        
        # Power loss penalty
        if power_difference < 0:
            revenue -= self.power_loss_penalty
            power_loss = True

        # Maintanance costs
        if action[0] > 0:  # ess1 cost
            revenue -= self.ess_cost
        if action[1] > 0:  # ess2 cost
            revenue -= self.ess_cost

        # Generator cost
        if generator_power > 0:
            revenue -= self.generator_cost * generator_power

        return revenue, power_loss