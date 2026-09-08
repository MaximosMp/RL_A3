"""Experiment driver for the report.

usage: python experiments.py part1
       python experiments.py part2 {size | speed | observation | speed-size}

part1 sweeps the four agent variants (MC, MC+baseline, n-step, n-step+baseline);
part2 varies one environment property at a time against the n-step+baseline
default. See exp.sh for the full set of runs behind the report.
"""
import sys
import time

import numpy as np

from Helper import write_to_doc
from REINFORCE_semi import reinforce

USAGE = __doc__.split("usage: ", 1)[1].rstrip()
SUBSECTIONS = ('size', 'speed', 'observation', 'speed-size')

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
if not args:
    raise SystemExit(f"usage: {USAGE}")

section = args[0]
subsection = args[1] if len(args) > 1 else None

if section not in ('part1', 'part2'):
    raise SystemExit(f"unknown section {section!r}\n\nusage: {USAGE}")
if section == 'part2' and subsection not in SUBSECTIONS:
    raise SystemExit(
        f"part2 needs one of {SUBSECTIONS}, got {subsection!r}"
        f"\n\nusage: {USAGE}")

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
                    f"\n\n === Running size = {rows}x{columns}, Rep.{j} === "
                    f"\n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                write_to_doc(
                    f'\n\n {stamp}, part2-size,{j},{rows}x{columns} ... '
                    f'params: {reinforce.params}, '
                    f'Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 2 - Speed Variation

    elif subsection == 'speed':
        speeds = [0.5,1.5,2.]
        for speed in speeds:
            for j in range(5):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running speed = {speed}, Rep.{j} === "
                    f"\n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                write_to_doc(
                    f'\n\n {stamp}, part2-speed,{j},{speed} ... '
                    f'params: {reinforce.params}, '
                    f'Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 3 - Observation Type
    elif subsection == 'observation':
        obs_types = ['vector']
        for obs_type in obs_types:
            for j in range(2):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running observation = {obs_type}, Rep.{j} === "
                    f"\n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                write_to_doc(
                    f'\n\n {stamp}, part2-observation,{j},{obs_type} ... '
                    f'params: {reinforce.params}, '
                    f'Avg reward: {np.mean(rewards):.3f} \n')

    # Experiment 4 - Environment - Speed variation
    elif subsection == 'speed-size':
        rows = 7
        columns = 9
        speeds = [0.5,1.]

        for speed in speeds:
            for j in range(5):
                stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))
                print(
                    f"\n\n === Running speed-size = {speed}, Rep.{j} === "
                    f"\n Stamp: {stamp} \n\n")
                rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                    max_misses, max_steps, seed, n_step, speed, boot,
                                    P_weights, V_weights, minibatch, eta, stamp, baseline)

                write_to_doc(
                    f'\n\n {stamp}, part2-speed-size,{j},{speed},size ... '
                    f'params: {reinforce.params}, '
                    f'Avg reward: {np.mean(rewards):.3f} \n')
    # Experiment 4 - other interesting variations
