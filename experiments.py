from REINFORCE_semi import reinforce
import time
import numpy as np
import sys

args = sys.argv[1:]

# game settings
n_episodes = 300
learning_rate = 0.01
rows = 7
columns = 7
max_misses = 10
max_steps = 250
n_step = 5
speed = 1.0
minibatch = 1
P_weights = None
V_weights = None
eta = 0.0005
seed = 13
obs_type = 'pixel'

# good parameter run -> plot
boot = ["MC","MC","n_step","n_step"]
baseline = [False, True, False, True]
i = int(args[0])  # we could also write a very short shell file that only takes an integer as CLI
# python experiments.py 1
# python experiments.py 2
#to run the experiments separately run "exp.sh"

# or just change it back to the way it was
# for i in [0,1,2,3]:

for j in range(5):
    stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
    print(f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")

    rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                        max_misses, max_steps, seed, n_step, speed, boot[i], 
                        P_weights, V_weights, minibatch, eta, stamp, baseline[i])

    with open("data/documentation.txt", 'a') as f:
            f.write(
                f'\n\n {stamp}, Exp{i},{j} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')

boot = "n_step"
baseline = True
# Experiment 1 - Size Variation
# square 7x7, square 14x14, rectangle 7x14, rectangle 14x7
list_of_rows_columns = [(7, 7), (14, 14), (7, 14), (14, 7)]


for j in range(5):
    stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
    print(f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
    for rows, columns in list_of_rows_columns:
        rows = list_of_rows_columns[i][0]
        columns = list_of_rows_columns[i][1]
        rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                            max_misses, max_steps, seed, n_step, speed, boot[i], 
                            P_weights, V_weights, minibatch, eta, stamp, baseline[i])

    with open("data/documentation.txt", 'a') as f:
            f.write(
                f'\n\n {stamp}, Exp{i},{j} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')



# Experiment 2 - Speed Variation
# velocity 0.5, 1.0, 1.5, 2.0
speeds = [0.5, 1.0, 1.5, 2.0]
for j in range(5):
    stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
    print(f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
    for speed in speeds:
        rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                            max_misses, max_steps, seed, n_step, speed, boot[i], 
                            P_weights, V_weights, minibatch, eta, stamp, baseline[i])

    with open("data/documentation.txt", 'a') as f:
            f.write(
                f'\n\n {stamp}, Exp{i},{j} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')

# Experiment 3 - Observation Type
obs_types = ['vector', 'pixel']

#Experiment 4 - Environment - Speed variation
rows = 7
columns = 14
speeds = [0.5, 1.0, 2.0]