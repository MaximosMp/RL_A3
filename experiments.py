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
boot = ["MC", "MC", "n_step", "n_step"]
baseline = [False, True, False, True]
# we could also write a very short shell file that only takes an integer as CLI
section = int(args[0])
i = int(args[1])

'''
python experiments.py 1
python experiments.py 2
to run the experiments separately run "exp.sh"

or just change it back to the way it was
for i in [0,1,2,3]:
'''
if section == 0:
    for eta in [0.0005, 0.001, 0.005, 0.01]:
        for j in range(1):
            stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
            print(
                f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")

            rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                max_misses, max_steps, seed, n_step, speed, boot[i],
                                P_weights, V_weights, minibatch, eta, stamp, baseline[i])

            with open("data/documentation.txt", 'a') as f:
                f.write(
                    f'\n\n {stamp}, Exp{i},{j} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')


# PART 2
if section == 1:

    boot = "n_step"
    baseline = True
    training = True
    # P_weights = 'data/weights/w_P_26_171012.h5'
    # V_weights = 'data/weights/w_V_26_171012.h5'

    # Experiment 0 - Size Variation
    # square 7x7, square 11x11, rectangle 7x14, rectangle 14x7
    if i == 0:
        list_of_rows_columns = [(11, 11), (7, 14), (14, 7)]

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

    # Experiment 1 - Speed Variation
    # velocity 0.5, 1.5, 2.0
    # but also include speed=1.0 in plot

    if i == 1:
        speeds = [0.5, 1.5, 2.0]
        for speed in speeds:
            for j in range(5):
                if j == 4:
                    continue
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{speed} ... params: {reinforce.params}, Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 2 - Observation Type
    # but also include 'pixel' in plot
    if i == 2:
        obs_types = ['vector']
        for obs_type in obs_types:
            for j in range(5):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running Experiment No.{i}, Rep.{j} === \n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                with open("data/documentation.txt", 'a') as f:
                    f.write(
                        f'\n\n {stamp}, Exp{i},{j},{obs_type} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')

    # Experiment 3 - Environment - Speed variation
    # but also include speed 1.0 (7x14) from 1st experiment
    if i == 3:
        rows = 7
        columns = 14
        speeds = [0.5, 2.0]

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
                        f'\n\n {stamp}, Exp{i},{j} ... params: {reinforce.params}, Avg reward: {np.mean(rewards)} \n')
    # Experiment 4 - other interesting variations
