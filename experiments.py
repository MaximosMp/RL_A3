from REINFORCE_semi import reinforce
import time
import numpy as np
import sys
from Helper import write_to_doc

args = sys.argv[1:]

# game settings
n_episodes = 400
learning_rate = 0.01
rows = 7
columns = 7
max_misses = 10
max_steps = 250
n_step = 5
speed = 1.0
minibatch = 4
P_weights = None
V_weights = None
eta = 0.001
seed = np.random.randint(100)
obs_type = 'pixel'


# good parameter run -> plot
boot = ["MC", "MC", "n_step", "n_step"]
baseline = [False, True, False, True]
# we could also write a very short shell file that only takes an integer as CLI
section = args[0]
subsection = args[1]

if section == 'part1':
    # here, "i" (second comand line argument) decides which experiment is: run MC, MC+baseline, Nstep, Nstep+baseline
    for i in range(4):
        # agent loop
        for j in range(5):
            # repitition loop
            stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
            print(
                f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp}")
            write_to_doc(f'\nExp{section},{i},{j}')
            rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                max_misses, max_steps, seed, n_step, speed, boot[i],
                                P_weights, V_weights, minibatch, eta, stamp, baseline[i])


# PART 2
# all experiments will be compared to the default 7x7, 1.0 speed, pixel etc.
if section == 'part2':
    boot = "n_step"
    baseline = True
    training = True

    # Experiment 1 - Size Variation
    if subsection == 'size':
        list_of_rows_columns = [(7,9),(9, 7),(9,9)]

        for rows, columns in list_of_rows_columns:
            for j in range(5):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{rows}x{columns} ... params: {reinforce.params}, Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 2 - Speed Variation

    elif subsection == 'speed':
        speeds = [0.5,1.5,2.]
        for speed in speeds:
            for j in range(5):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{speed} ... params: {reinforce.params}, Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 3 - Observation Type
    elif subsection == 'observation':
        obs_types = ['vector']
        for obs_type in obs_types:
            for j in range(2):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{obs_type} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')

    # Experiment 4 - Environment - Speed variation
    elif subsection == 'speed-size':
        rows = 7
        columns = 9
        speeds = [0.5,1.]

        for speed in speeds:
            
            for j in range(1):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{speed},size ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')
    # Experiment 4 - other interesting variations
