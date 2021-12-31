# User defined libraries
# from config import *
import pandas as pd

from controllers import Agent, DQN
from environment import HRES

from utils import plot_total_rewards, Logger, set_seed
import itertools

def main():
    # Hyper Parameter lists
    batch_sizes = [50]
    memory_sizes = [5000] 

    lrs = [1e-3, 2e-3, 5e-3]
    learn_everys = [10, 20, 30]
    taus = [1e-3, 3e-3, 5e-3]
    gammas = [0.9]
    eps_start = 1
    eps_end = 0.1
    eps_decay = 150e3
    seed = 42


    data_path = './data/'

    # Perform tuning
    tune_parameters(data_path, batch_sizes, memory_sizes, lrs, learn_everys, gammas, eps_start, eps_end, eps_decay, taus, seed)




def tune_parameters(data_path, batch_sizes, memory_sizes, lrs, learn_everys, gammas, eps_start, eps_end, eps_decay, taus, seed):
    
    set_seed(42)

    # Initialize the environment
    env = HRES(data_path, mode='train')

    # Episodes
    ep_num = 50

    # Results
    results = pd.DataFrame(columns=['batch_size', 'memory_size', 'lr', 'tau', 'learn_every', 'sync_every', 'gamma', 'eps_decay', 'max_reward', 'last_reward'])
    path_to_results = 'drive/MyDrive/ACSE-9_data/'

    print('Starting Hyper Search...')
    num = 0

    # Loops through the specified
    for batch_size, memory_size, lr, tau, learn_every, gamma, in itertools.product(batch_sizes, memory_sizes, lrs, taus, learn_everys, gammas): 

        print('Tunning ep: ', num)
        print('Config:  lr={}, tau={}, learn_every={}'.format(lr, tau, learn_every))

        # Create new agent
        agent = Agent(batch_size=batch_size, memory_size=memory_size, lr=lr, tau=tau,
                        learn_every=learn_every, gamma=gamma, 
                            eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)


        # ************* TRAINING LOOP ***********************

        total_rewards = []
        max_reward = -10e6  # some large negative number

        # Loop over the specified number of episodes
        for ep in range (ep_num):

            # reset the env at the begining of each episode
            env.reset()

            # Make the initial state observation
            state = env.observe()

            for time in range(env.data_size-1):    

                # Select the action based on epsilon-greedy policy
                action = agent.select_action(state)

                # Based on the selected action, perform the temporal action step within the environment
                reward, next_state = env.step(action)

                # Perform the temporal step in agent experience sampling and training
                loss = agent.step(state, action, reward, next_state)

                # Make the next state the current state and repeat
                state = next_state

            # Append the accumulated training total rewards to the tracker
            total_rewards.append(env.total_rewards)
            
        max_reward = max(total_rewards)

        last_reward = env.total_rewards
        config = {'batch_size':batch_size, 'memory_size':memory_size, 'lr':lr, 'tau':tau, 'learn_every':learn_every, 'sync_every':sync_every, 
                  'gamma': gamma, 'eps_decay':eps_decay, 'max_reward':max_reward, 'last_reward': last_reward}

        results = results.append(config, ignore_index=True)
        num += 1

    # Save results to csv
    print('Saving results to csv!')
    results.to_csv(path_to_results + 'tune_results_{}_eps.csv'.format(ep_num))

if __name__ == "__main__":
    main()
