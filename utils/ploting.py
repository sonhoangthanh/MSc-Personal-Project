import numpy as np
import matplotlib.pyplot as plt

# Heler function to plot the input rewards array over the number of episodes
def plot_total_rewards(total_rewards, num_episodes):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    # Plot time
    ax.plot(np.arange(0, num_episodes), total_rewards, linewidth=5)
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Total Reward')
    
    plt.show()