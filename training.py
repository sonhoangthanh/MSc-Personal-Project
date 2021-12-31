# User defined libraries
from controllers import Agent, DQN
from environment import HRES

from utils import plot_total_rewards, Logger, set_seed


def main():

    # Hyper Parameters
    batch_size = 50
    memory_size = 5000
    lr = 1e-3
    learn_every = 20
    gamma = 0.9
    eps_start = 1 
    eps_end = 0.1
    eps_decay = 150000
    tau = 1e-3
    seed = 42
    set_seed(42)

    # Paths
    save_path = 'saved_models/trained_model'
    data_path = './data/'

    # Initialize the environment
    env = HRES(data_path, reward='revenue')

    agent = Agent(batch_size=batch_size, memory_size=memory_size, lr=lr, learn_every=learn_every,
                                                        gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
    ep_num = 80

    # Perform training
    total_rewards, logger = training(env, agent, ep_num, save_path)



def training(env, agent, ep_num, save_path):

    logger = Logger()

    print('Starting Training')

    # ************* TRAINING LOOP ***********************

    total_rewards = []

    # Loop over the specified number of episodes
    for ep in range (ep_num):

        print('ep = {}'.format(ep))
        # reset the env at the begining of each episode
        env.reset()

        # Make the initial state observation
        state = env.observe()

        for time in range(env.data_size-1):    

            # Select the action based on epsilon-greedy policy
            action = agent.select_action(state)

            # Based on the selected action, perform the temporal action step within the environment
            reward, next_state = env.step(action)

            # Perform the temporal step in agent's experience sampling and training
            loss = agent.step(state, action, reward, next_state)

            logger.push_log_train(loss)

            # Make the next state the current state and repeat
            state = next_state

        # Append the accumulated training total rewards to the tracker
        total_rewards.append(env.total_rewards)

    print('Saving model...')
    # Save the model's state dictionary 
    agent.save(save_path)

    # Print the maximum training reward
    print('Maximum episode reward = ', max(total_rewards))

    return total_rewards, logger



# Execute if called as the main program
if __name__ == '__main__':
    main()
