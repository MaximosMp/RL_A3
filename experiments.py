from REINFORCE_semi import reinforce
import time
import numpy as np


# game settings
n_episodes = 500
learning_rate = 0.01
rows = 7
columns = 7
max_misses = 10
max_steps = 250
n_step = 5
speed = 1.0
minibatch = 1
weights = None
eta = 0.01
seed = 13
obs_type = 'vector'

# good parameter run -> plot
boot = ['MC', 'n_step', 'n_step']
baseline = [False, False, True]

for i in [0,1,2]:
    

    for j in range(5):
        start = time.time()
        stamp = time.strftime("%d_%H%M%S", time.gmtime(start))
        
        rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                            max_misses, max_steps, seed, n_step, speed, boot[i], 
                            weights, minibatch, eta, stamp, baseline[i])

        with open("data/documentation.txt", 'a') as f:
                f.write(
                    f'\n\nStamp: {stamp} ... baseline: {baseline[i]}, boot: {boot[i]}, Avg reward: {np.mean(rewards)} \n')
        
                
    