#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_and_mean(files: list[str]) -> list:
    rewards = []
    for plot in files:
        rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
    return np.mean(rewards,axis=0)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']


### Part 1

part1_full = ['27_220351','27_230853','28_002357','28_013246','28_065433']
part1_reinforce = ['27_182504','27_184923','27_191045','27_193144','27_195244']
part1_bootstrap = ['27_201503','27_205749','28_075635']
part1_MCbaseline = ['28_153349']

hyperopt_big = ['21_205207','21_224331','21_205312','22_002411','22_015841','22_021350','22_042033','22_054051','22_063949']
hyperopt_big += ['22_072616','22_083833','22_101711','22_094037','22_140345','22_163358','22_180822','22_194411','22_211909']
hyperopt_big += ['22_231129','23_004433']

learning_plots = ['']
eta_plots = ['25_181333','25_201320','25_221342','26_151751','26_171012','26_175224']


rewards = []
for plot in part1_reinforce:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label='REINFORCE')


rewards = []
for plot in part1_MCbaseline:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label = 'MC baseline')


rewards = []
for plot in part1_bootstrap:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label='bootstrap')


rewards = []
for plot in part1_full:
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
rewards = np.mean(rewards,axis=0)
plt.plot(savgol_filter(rewards, 10,1), label = 'full')

plt.title('End of Part 1')
plt.grid()
plt.legend()
plt.show()

for j, plots in enumerate(hyperopt_big):
    linestyle = "-"
    if j > 9:
        linestyle = "--"
    labl = f'{j}:' + plots
    reward = np.load(f'data/rewards/r_{plots}.npy')
    reward = savgol_filter(reward,10,1)
    plt.plot(reward, label=labl, linestyle=linestyle)
plt.title('Grid hyperopt')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(fontsize='8')
plt.show()

for j, plots in enumerate(eta_plots):
    linestyle = "-"
    labl = f'{j}:' + plots
    reward = np.load(f'data/rewards/r_{plots}.npy')
    reward = savgol_filter(reward,10,1)
    plt.plot(reward, label=labl, linestyle=linestyle)
plt.title('Eta hyperopt')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()


### PART 2
default = ['30_202045','30_204133','30_210223','30_212339','01_065527']
speed_05 = ['30_113517','30_115449','30_121431','30_123431','30_140008']
speed_15 = ['30_151006','30_151743','30_152617','30_153421','30_142345']
speed_20 = ['30_154243','30_154719','30_155202','30_155709']

plt.plot(savgol_filter(load_and_mean(default), 10,1), label = 'default')
plt.plot(savgol_filter(load_and_mean(speed_05), 10,1), label = 'speed 0.5')
plt.plot(savgol_filter(load_and_mean(speed_15), 10,1), label = 'speed 1.5')
plt.plot(savgol_filter(load_and_mean(speed_20), 10,1), label = 'speed 2.0')

plt.ylim(-10,36)
plt.title('Default vs speed 0.5')
plt.grid()
plt.legend()
plt.show()
