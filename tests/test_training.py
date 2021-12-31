# # Imports
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# # Pytorch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random

# # User defined libraries
# # from config import *
# from controllers import Agent, DQN
# from environments import HRES

# from utils import plot_total_rewards, Logger, set_seed

# def test_training():

#     # Hyper Parameters
#     batch_size = 50
#     memory_size = 4000

#     lr = 1e-3
#     learn_every = 5
#     sync_every = 300
#     gamma = 0.99
#     eps_start = 0.99
#     eps_end = 0.01
#     eps_decay = 500

#     seed = 42

#     set_seed(42)

#     data_path = './data/'
#     model_path = './saved_models/trained_model'


#     # Initialize the environment
#     env = HRES(data_path, mode='eval')

#     # print(env.data_size)

#     agent_apply = Agent(mode='eval', saved_path=model_path)

#     # Test the random controller

#     random_reward = 0
#     model_reward = 0
#     env.reset()
#     state = env.observe()
    
#     for time in range(env.data_size - 1):    

#         # Select the action based on epsilon-greedy policy
#         action_random = random.choice(np.arange(0, 18))

#         # Based on the selected action, perform the temporal action step within the environment
#         reward, next_state = env.step(action_random)

#         # Make the next state the current state and repeat
#         state = next_state

#         random_reward += reward

#     print(random_reward)

#     env.reset()
#     state = env.observe()

#     for time in range(env.data_size - 1):
#         # Select the action based on epsilon-greedy policy
#         action = agent_apply.select_action(state)

#         # print(action)

#         # Based on the selected action, perform the temporal action step within the environment
#         reward, next_state = env.step(action)

#         # Make the next state the current state and repeat
#         state = next_state

#         model_reward += reward
    
#     print(model_reward)
