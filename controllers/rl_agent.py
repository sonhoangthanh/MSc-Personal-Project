# This is the file for reinforcement agent

import numpy as np
from collections import deque, namedtuple
import random
import math
import copy

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .memory import ReplayMemory
from .DQN import DQN
from utils import Logger


class Agent():
    """ 
        Implements RL Agent based on Double DQN

        Inputs: 
            (int) observation_size - size of observation space
            (int) action_size - size of action space
            (int) batch_size - size of minibatch
            (int) memory_size - size of memory buffer
            (float) lr - learning rate
            (float) tau - soft update factor
            (float) gamma - discount factor
            (float) mode - can be either 'train' or 'eval'
            (float) saved_path - path to save the trained model state dictonary
            (float) eps_start - initial epsilon
            (float) eps_end - final epsilon
            (float) eps_decay - epsilon value decay
            (int) learn_every - learning frequence
    """

    def __init__(self, observation_size=5, action_size=18,batch_size=50, memory_size=2000, lr=1e-3, tau=1e-3, gamma=0.90, mode='train', 
                                saved_path=None, eps_start=1, eps_end=0.1, eps_decay=5e4, learn_every=10):
        
        # Enable CUDA acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.observation_size = observation_size
        self.action_size = action_size

        if self.mode == 'train':
            # Q-Network
            self.policy_net = DQN(observation_size, action_size).to(self.device)
            self.target_net = copy.deepcopy(self.policy_net).to(self.device)

            # Freeze the parameters in the target network
            for param in self.target_net.parameters():
                param.requires_grad = False

            # Epsilon-Greedy rate
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay

            self.gamma = gamma
            self.tau = tau

            # Q-learning
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
            self.loss = torch.nn.SmoothL1Loss()
            self.batch_size = batch_size
            self.memory = ReplayMemory(self.batch_size, capacity=memory_size)
            self.steps_done = 0

            self.min_exp = 5 * self.batch_size  # min number of experiences before agent learns
            self.learn_every = learn_every


        elif self.mode == 'eval' and saved_path != None:
            # Create the model and load the pretrained state dictionary
            self.model = DQN(observation_size, action_size).to(self.device)
            self.model.load_state_dict(torch.load(saved_path, map_location=self.device))
            self.model.eval()

        else:
            raise('Incorrect mode!')

        
    # Function which will select the action based on the epsilon-greedy method
    def select_action(self, state):
        """ 
            Returns action for a given state given the current policy. Method uses decaying epsilon-greedy method.
        """
        # Create state tensor
        state = torch.from_numpy(state).float().to(self.device)
    
        # If in evaluation mode, only propagate through the model and obtain the action index
        if self.mode == 'eval':
            with torch.no_grad():
                # Forward pass the state through the network
                result = self.model(state)

                # Obtain the action idex through the maximum of the estimated Q-values
                action_index = torch.argmax(result).cpu().numpy()
                return action_index

        # Calculate the epsilon value with decay, for every eps_decay steps, the epsilon decays by factor of 1 / e
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        # eps_threshold = 0.2

        # Based on the sample, do either exploration (sample > epsilon) or exploitation
        sample = random.random()

        # Initialize
        action_index = 0
        
        # Exploit the existing policy network
        if sample > eps_threshold:
            with torch.no_grad():
                # Forward pass the state through the network
                result = self.policy_net(state)

                # Obtain the action idex through the maximum of the estimated Q-values
                action_index = torch.argmax(result).cpu().numpy()
        else:
            # Explore using random sampling from the action space
            action_index = random.choice(np.arange(0, self.action_size))
            
        return action_index


    # Estimate of the Q-value for the current state
    def td_estimate(self, states, actions):
        current_Q = self.policy_net(states).gather(1, actions)
        return current_Q.squeeze()

    # Estimation of the the target Q-value from the Bellman's Equation for optimal policy
    def td_target(self, rewards, next_states):
        # Obtain the Q-values for the next stake achieved with action taken by the current, best policy
        next_state_Qs = self.policy_net(next_states)

        # Find the best action based on the policy Q-values, resize so they could be used for indexing 
        best_actions = torch.argmax(next_state_Qs, axis=1).resize_(self.batch_size, 1)

        # Find the Q-value of the next state indexed for batch and best action for current policy
        next_Qs = self.target_net(next_states).gather(1, best_actions)
        target_Qs = rewards + self.gamma * next_Qs
        return target_Qs.squeeze()


    # Function to perform the soft updates between the policy and target networks
    def soft_update(self, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            policy_net (PyTorch Neural Network): weights will be copied from
            target_net (PyTorch Neural Network): weights will be copied to
            tau (float): interpolation parameter to update the weights between both networks
        """
        # Perform the soft updates on the parameters of the policy and target networks
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)


    def step(self, state, action, reward, next_state):

        # Push the experience to the memory buffer 
        self.memory.push(state, action, reward, next_state)

        # Learn every UPDATE_EVERY time steps.
        self.steps_done += 1

        # Initialize
        loss = None

        if self.steps_done % self.learn_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.min_exp:
                experiences = self.memory.sample()
                loss = self.learn(experiences)

        return loss


    def learn(self, experiences):
        """ Functions which performs Q-learning on the batch experiences with gamma discount rates. """

        # Unpack the experiences
        states, actions, rewards, next_states = experiences

        # Calculate the expected and the target Q values from policy and target networks
        Q_expected = self.td_estimate(states, actions)

        Q_targets = self.td_target(rewards, next_states)

        # Compute the loss function
        loss = self.loss(Q_expected, Q_targets)

        # Set the gradients of optimizer to zero
        self.optimizer.zero_grad()

        # Do the backwards propagation
        loss.backward()
        
        # Step using the optimizer and update using gradient descent
        self.optimizer.step()

        # Perform a soft update on target network model parameters from the policy network
        self.soft_update(self.tau)

        return loss.item()

    # Function to save the state dictionary of the policy network
    def save(self, save_path):
        if save_path is not None:
            torch.save(self.policy_net.state_dict(), save_path)