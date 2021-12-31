import pytest
import random

from controllers import ReplayMemory


""" Test for ReplayMemory """

def test_memory():
    random.seed(42)
    
    memory_size = 10
    batch_size = 10

    memory = ReplayMemory(batch_size, memory_size)
    # memory.push(0, 0, 0, 0)

    # assert len(memory) == 1

    # Push some random values
    for i in range(20):
        state = (random.random(), random.random(), random.random(), random.random())
        action = random.randint(0, 6)
        reward = random.randint(0, 15)
        next_state = (random.random(), random.random(), random.random(), random.random())

        memory.push(state, action, reward, next_state)

    # Check the length after the update
    assert len(memory) == 10

    # Test sampling
    states, actions, rewards, next_states = memory.sample()

    assert len(states) == batch_size
    assert len(actions) == batch_size
    assert len(rewards) == batch_size
    assert len(next_states) == batch_size
