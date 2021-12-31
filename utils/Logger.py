import numpy as np
import matplotlib.pyplot as plt

# Class object which records all the 
class Logger:
    def __init__(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.total_rewards = []
            self.loss_values = []
        elif mode == 'eval':
            self.rewards = []
            self.total_supply_power = []
            self.solar_power = []
            self.wind_power = []
            self.load = []
            self.power_ess1 = []
            self.power_ess2 = []
            self.power_generator = []
            self.ess1_charge_states = []
            self.ess2_charge_states = []
            self.power_loss = 0
        else:
            raise('Set correct mode!')

    def reset(self):
        if self.mode == 'train':
            self.total_rewards.clear()
            self.loss_values.clear()
        elif self.mode =='eval':
            self.rewards.clear()
            self.total_supply_power.clear()
            self.solar_power.clear()
            self.wind_power.clear()
            self.load.clear()
            self.power_ess1.clear()
            self.power_ess2.clear()
            self.power_generator.clear()
            self.ess1_charge_states.clear()
            self.ess2_charge_states.clear()
            self.power_loss = 0

    def push_log_train(self, loss):
        if loss == None:
            return
        else:
            self.loss_values.append(loss)
 

    def push_log_eval(self, reward, total_supply_power, solar_power, wind_power, load, 
                                    power_ess1, power_ess2, power_generator, ess1_charge_states, ess2_charge_states, power_loss):
        self.rewards.append(reward)
        self.total_supply_power.append(total_supply_power)
        self.solar_power.append(solar_power)
        self.wind_power.append(wind_power)
        self.load.append(load)
        self.power_ess1.append(power_ess1)
        self.power_ess2.append(power_ess2)
        self.power_generator.append(power_generator)
        self.ess1_charge_states.append(ess1_charge_states)
        self.ess2_charge_states.append(ess2_charge_states)

        if power_loss == True:
            self.power_loss += 1


    def plot_loss(self):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        ax.plot(np.arange(0, len(self.loss_values)), self.loss_values, label='Loss value')
        ax.set_title('Loss values')

        plt.show()


    def plot_eval_all(self):
        if (self.mode == 'train'):
            return

        time = np.arange(0, len(self.rewards))
        
        fig1 = plt.figure(figsize=(15, 10))
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure(figsize=(15, 10))
        ax2 = fig2.add_subplot(111)

        fig3 = plt.figure(figsize=(15, 10))
        ax3 = fig3.add_subplot(111)

        # Plot the rewards over time
        ax1.plot(time, self.rewards)
        ax1.set_xlabel('Time [h]')
        ax1.set_ylabel('Reward value')

        # Plot total supply power vs load
        ax2.plot(time, self.load, label='Load Power')
        ax2.plot(time, self.total_supply_power, label='Total Supply Power')
        ax2.set_xlabel('Time [h]')
        ax2.set_ylabel('Total Supply Power [kW]')
        ax2.legend(loc='best')

        # Plot the charge state of the ess modules
        ax3.plot(time, self.ess1_charge_states, label='battery energy state')
        ax3.plot(time, self.ess2_charge_states, label='hydrogen energy state ')
        ax3.set_xlabel('Time [h]')
        ax3.set_ylabel('Energy [MWh]')
        ax3.legend(loc='best')


        print('Number of power_losses = {}'.format(self.power_loss))
        plt.show()

