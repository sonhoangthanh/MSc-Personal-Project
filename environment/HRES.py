# Imports
import math
import gym
from gym.utils import seeding
from gym.spaces import Box

import numpy as np
import matplotlib.pyplot as plt

from storage import simple_battery, simple_hydrogen, generator
from utils import load_data
from utils import Logger
from rewards import FinancialModel

class HRES():
    """ 
    Description:
        An environment model of the Hybrid Renewable Energy Power Plant (HRES) [Solar + Wind] with hybrid Energy Storage System (ESS). Solar PV and wind turbines generate power 
        with data read to the class at initialization including the load data.py

        Observable state:
        - solar generated power in kW
        - wind generated power in kW
        - load data in kW
        - charge state of ESS 1
        - charge state of ESS 2

        Available Actions:
        - 3 element action array consisting of: do nothing, charge , discharge actions for both ess1 and ess2 modules, generator off/on,
            thus resulting in 18 distinct combinations and actions.

        Inputs: 
                (string) datapath - path to solar, wind and load data
                (string) mode - can be either 'train' or 'eval'
                (string) reward - by default 'revenue', can be made to point to different reward functions in the future
    """

    # Initialization function
    def __init__(self, data_path, mode='train', reward='revenue'):
        # Set the mode to either training or application
        self.mode = mode
        
        # Input files
        # Load the power data
        self.solar_power, self.wind_power, self.load = load_data(data_path, self.mode)
        self.data_size = len(self.load)
        
        # Total power
        # self.hybrid_power = self.solar_power + self.wind_power
        
        # Energy storage system, initialized with default values
        self.ess1 = simple_battery()
        self.ess2 = simple_hydrogen()

        # Backup generator
        self.generator = generator()

        # Action space
        self.action_space = [[ess1, ess2, gen] for ess1 in [-self.ess1.power, 0, self.ess1.power] for ess2 in [-self.ess2.power, 0, self.ess2.power] for gen in range(2)]

        # Reward function
        self.reward_type = reward
        if self.reward_type == 'revenue':
            self.reward_model = FinancialModel()  # initialize with default values
        else:
            raise Exception('Incorrect mode!')

        # Timestep counter
        self.time = 0
        
        # Flag to determine if the time duration for training is done
        self.is_done = False
        
        self.state = None

        self.total_rewards = 0

        # Data Logger
        if self.mode == 'eval':
            self.logger = Logger(self.mode)
            self.reward_record = []
    

    # **************** UTILS *******************************
    
    # Function to set the random seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    # Function which resets the state of the environment to initial values at t=0
    def reset(self):
        self.ess1.energy_state = 200e3
        self.ess2.energy_state = 200e3
        self.time = 0
        self.total_rewards = 0
        self.reward_record = []
        
        if self.mode == 'eval':
            self.logger.reset()

        # reset and initialize the first state
        self.state = np.array([self.solar_power[self.time], self.wind_power[self.time], self.load[self.time], 
                                        self.ess1.energy_state, self.ess2.energy_state], dtype=np.float32)
    
    
    # ***************** STATE TRANSITIONS ***********************

    
    # Function which return the current state of the environment
    def observe(self):
        return np.array([self.solar_power[self.time], self.wind_power[self.time], self.load[self.time], 
                                        self.ess1.energy_state, self.ess2.energy_state], dtype=np.float32)


    def perform_action(self, action_index):

        # From the input action index, get the action array from avaialble action space
        action = self.action_space[action_index]

        # Positive power is dicharge, negative power is charge
        power_ess1 = 0
        power_ess2 = 0
        power_generator = 0

        if action[0] == -self.ess1.power:  # charge ess1
            power_ess1 = self.ess1.charge()
        elif action[0] == self.ess1.power:  # discharge ess1
            power_ess1 = self.ess1.discharge()
        else:  # ess1 does nothing
            pass

        if action[1] == -self.ess2.power:  # charge ess2
            power_ess2 = self.ess2.charge()
        elif action[1] == self.ess2.power:  # discharge ess2
            power_ess2 = self.ess2.discharge()
        else:  # ess2 does nothing
            pass
        
        if action[2] == 1:
            power_generator = self.generator.generate_power()


        # ************* REWARDS **************

        reward = 0
        power_loss = False

        total_supply_power = self.solar_power[self.time] + self.wind_power[self.time] + power_ess1 + power_ess2 + power_generator
        
        if self.reward_type == 'revenue':  # financial reward
            reward, power_loss = self.reward_model.get_reward(total_supply_power, self.load[self.time], action, power_generator)

        
        # If in evaluation mode, append to the logger object
        if self.mode == 'eval':
            # Log the data
            self.logger.push_log_eval(reward, total_supply_power, self.solar_power[self.time], self.wind_power[self.time],
                                        self.load[self.time], power_ess1, power_ess2, power_generator, self.ess1.energy_state, self.ess2.energy_state, power_loss)

        return reward

    # Step function, which will update the environment state and returns the reward for the action
    def step(self, action):
        
        # Perform the action
        reward = self.perform_action(action)

        # Update time counter
        self.time += 1

        # Observe the next state
        next_state = self.observe()

        if self.mode == 'eval':
            self.reward_record.append(reward)
        
        # Add the reward to the cumulative pool
        self.total_rewards += reward 
        
        return reward, next_state
        

    # Function to plot the evolution of the environement e.g. power delivered
    # Time in hours
    def plot_data(self, time_start, time_end):
        
        time_duration = time_end - time_start
        # demand_precision = 12  # 5 min precision for demand

        # Get data
        wind_data = self.wind_power[time_start:time_end]
        solar_data = self.solar_power[time_start:time_end]  # need to multiply by 12 such that 5 min data can be transformed to hourly
        load_data = self.load[(time_start):(time_end)]

        # Time data
        time = np.arange(time_start, time_end, 1)

        fig1 = plt.figure(figsize=(15, 10))
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure(figsize=(15, 10))
        ax2 = fig2.add_subplot(111)

        # Plot the data
        ax1.plot(time, wind_data, label='wind')
        ax1.plot(time, solar_data, label='solar')
        ax1.plot(time, load_data, label='demand')
        
        # Legend
        ax1.legend(loc='best')
        ax1.set_title('Data from {} h to {} h'.format(time_start, time_end))

        # Plot the difference between the renewable output and the demand
        deficit = np.array(wind_data) + np.array(solar_data)- np.array(load_data)


        # print(len(deficit))
        ax2.plot(time, deficit, label='deficit')
        ax2.set_title('Power deficit from {} h to {} h'.format(time_start, time_end))

        plt.show()

    def plot_rewards(self, end_time):

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

        # Plot time
        ax.plot(np.arange(0, end_time), self.reward_record)
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Reward')
        
        plt.show()

    

