# Imports
from controllers import Agent, RuleBasedControl, RandomControl
from environment import HRES
from utils import plot_total_rewards, set_seed, Logger
import copy

# Program which performs application of the trained controller onto the test dataset
def main():

    set_seed(42)

    data_path = './data/'
    model_path = './results/trained_model_revenue'

    # Initialize the environment the application environement
    env_apply = HRES(data_path, mode='eval', reward='revenue')

    logger_eval = apply_model(env_apply, data_path, model_path, type='rl')
    # Plot
    logger_eval.plot_eval_all()


    
def apply_model(env_apply, data_path, model_path, type='rl'):
    
    env = copy.deepcopy(env_apply)

    agent = None
    if type == 'rl':
        agent_apply = Agent(mode='eval', saved_path=model_path)
    elif type == 'random':
        agent_apply = RandomControl(len(env_apply.action_space))
    elif type == 'rule':
        agent_apply = RuleBasedControl(env_apply.ess1.capacity, env_apply.ess2.capacity, env_apply.ess1.power, env_apply.ess2.power, env_apply.action_space)

    print('Starting Evaluation...')

    env.reset()

    state = env.observe()

    rewards = 0
    logger = Logger()

    # rule_controller = RuleBasedControl(env_apply.ess1.capacity, env_apply.ess2.capacity, env_apply.ess1.power, env_apply.ess2.power, env_apply.action_space)

    for time in range(env_apply.data_size-1):    

        # Select the action based on epsilon-greedy policy
        action = agent_apply.select_action(state)

        # Based on the selected action, perform the temporal action step within the environment
        reward, next_state = env.step(action)

        # Make the next state the current state and repeat
        state = next_state

        rewards += reward

    print('total reward = ',rewards)
    logger = copy.deepcopy(env.logger)
    # At the end plot the results
    return logger

if __name__ == '__main__':
    main()