from controllers import Agent
from controllers import DQN
from environment import HRES
from utils import set_seed

import numpy as np
import pytest 
import torch


def test_agent():

    # Hyper Parameters
    batch_size = 5
    memory_size = 10

    lr = 1e-3
    learn_every = 10
    gamma = 0.90
    eps_start = 1
    eps_end = 0.05
    eps_decay = 500
    tau = 1e-3

    seed = 42

    set_seed(42)

    data_path = './data/'


    # Initialize the environment
    env = HRES(data_path, mode='train')

    # print(env.data_size)

    agent = Agent(batch_size=batch_size, memory_size=memory_size, lr=lr, tau=tau, learn_every=learn_every, 
                                                        gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

    # Test init
    assert agent.batch_size == batch_size
    assert agent.min_exp == 5 * batch_size
    assert agent.learn_every == learn_every
    assert agent.gamma == gamma
    assert agent.eps_start == eps_start
    assert agent.eps_end == eps_end
    assert agent.eps_decay == eps_decay
    assert agent.tau == tau

    # # Test action selection
    # env.reset()
    # state = env.observe()

    # action = agent.select_action(state)
    # assert action in np.arange(0, 18)

    # # Loop through 5 time iterations
    # for time in range(5):

    #     # Select the action based on epsilon-greedy policy
    #     action = agent.select_action(state)

    #     # Based on the selected action, perform the temporal action step within the environment
    #     reward, next_state = env.step(action)

    #     # Perform the temporal step in agent experience sampling and training
    #     loss = agent.step(state, action, reward, next_state)

    #     # Make the next state the current state and repeat
    #     state = next_state

    # states, actions, rewards, next_states = agent.memory.sample()

    # assert len(agent.memory) == 5
    # assert len(states) == 5
    # assert len(actions) == 5
    # assert len(rewards) == 5
    # assert len(next_states) == 5

    # # Check batch training method
    # # Propagate the input tensor
    # result = agent.policy_net(states)

    # assert len(result) == 5  # check that the returned tensor is the size of batch size which is 5
    # assert len(result[0]) == 18  # check that the each element in tensor is a tensor of size 18, the size of action space

    # # Get the maximum values from each row and resize into (5,1) array
    # chosen_actions = torch.argmax(result, axis=1).resize_((5, 1))
    # # Index the array, get the Q values corresponding to the maximum
    # chosen_result = result.gather(1, chosen_actions)
    # assert chosen_result.size() == (5,1)
