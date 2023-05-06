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

def load_list(files: list[str]) -> list:
    return np.array([np.load(f'data/rewards/r_{plot}.npy') for plot in files])

def plot_list(list_strings: list[str],label: str = None, color: str = None):
    rewards = load_and_mean(list_strings)
    plt.plot(savgol_filter(rewards,10,1),label=label, color=color)

def plot_list_errors(list_strings: list[str], save_name: str = None,color:str='#BEBEBE'):
    reward = load_list(list_strings)
    # Calculate the mean and standard deviation of the data
    means = np.mean(reward, axis=0)
    stds = np.std(reward, axis=0, ddof=1)
    # Calculate the standard error of the mean
    sem = stds / np.sqrt(reward.shape[1])
    sem = abs(sem)
    # Calculate the 95% confidence interval
    conf_int = 1.96 * sem

    # Plot the means with error bars representing the confidence interval
    # plt.errorbar(np.arange(reward.shape[1]), savgol_filter(means,10,1), yerr=conf_int, fmt='o', capsize=5,color='tab:red',markersize=0.2)
    plt.fill_between(np.arange(means.shape[0]),savgol_filter(means,10,1)-conf_int,savgol_filter(means,10,1)+conf_int,alpha=0.3,color=color)

def list_full_plot(list_strings: list[str], color:str='tab:red', label=None, save_name:str = None):
    plot_list(list_strings=list_strings,label=label,color=color)
    plot_list_errors(list_strings=list_strings,color=color)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']


### Part 1


part1_reinforce = ['04_215854']
part1_MCbaseline = ['04_225730','05_004942']
part1_bootstrap = ['05_034633','05_053914']
part1_full = ['04_183040','05_075841']

hyperopt_big = ['21_205207','21_224331','21_205312','22_002411','22_015841','22_021350','22_042033','22_054051','22_063949']
hyperopt_big += ['22_072616','22_083833','22_101711','22_094037','22_140345','22_163358','22_180822','22_194411','22_211909']
hyperopt_big += ['22_231129','23_004433']

learning_plots = ['23_131422','23_151619','23_203642','23_225132']  # with 0.01 eta
eta_plots = ['26_171012','26_151751','26_175224']  # with 0.001 learning '25_181333','25_201320','25_221342' more runs but bad

plot_list(part1_reinforce,'REINFORCE',color='tab:blue')
plot_list_errors(part1_reinforce,color='tab:blue')


plot_list(part1_MCbaseline,'MC baseline',color='tab:orange')
plot_list_errors(part1_MCbaseline,color='tab:orange')


plot_list(part1_bootstrap,'5-step bootstrap',color='tab:green')
plot_list_errors(part1_bootstrap,color='tab:green')


plot_list(part1_full,'full','tab:red')
plot_list_errors(part1_full,color='tab:red')

plt.title('End of Part 1')
plt.grid()
plt.legend()
plt.savefig('plots/Part_1.pdf')
plt.show()

# 0.5, 0.1, 0.01, 0.001, 0.0005, 0.0001
tuning = eta_plots + learning_plots

labels = [r'$\eta=0.001$',r'$\eta=0.0005$',r'$\eta=0.0001$',
          r'$\alpha=0.1$',r'$\alpha=0.01$',r'$\alpha=0.001$',r'$\alpha=0.0001$']
for j, plots in enumerate(tuning):
    linestyle = "-"
    if j > 2:
        linestyle = "--"
    reward = np.load(f'data/rewards/r_{plots}.npy')
    reward = savgol_filter(reward,10,1)
    plt.plot(reward, label=labels[j], linestyle=linestyle)
plt.title('Hyperparameter tuning')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='right')  # fontsize='8'
plt.savefig('plots/learning-eta_tuning.pdf')
plt.show()


### PART 2
speed_20_79 = ['06_000442','05_231413','05_171708']
speed_05_79 = ['05_123003']

list_full_plot(speed_05_79,label='speed 0.5, size 7x9',color='tab:blue')
list_full_plot(speed_20_79,label='speed 2.0, size 7x9',color='tab:orange')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.legend()
plt.savefig('plots/mixed_experiment.pdf')
plt.show()





nine_nine = ['04_210821','04_213900','04_221523','04_224709','04_232119']
speed_05 = ['05_002934','05_012051','05_021216']

list_full_plot(speed_05, label='speed = 0.5', color = 'tab:blue')
list_full_plot(nine_nine,label='shape = 9x9', color= 'tab:orange')

plt.title('some of part2')
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='best')  # fontsize='8'
plt.savefig('plots/part2.pdf')
plt.show()



default = ['30_202045','30_204133','30_210223','30_212339','01_065527']
speed_05 = ['30_113517','30_115449','30_121431','30_123431','30_140008']
speed_15 = ['30_151006','30_151743','30_152617','30_153421','30_142345']
speed_20 = ['30_154243','30_154719','30_155202','30_155709']

old_part1_full = ['27_220351','27_230853','28_002357','28_013246','28_065433']
old_part1_reinforce = ['27_182504','27_184923','27_191045','27_193144','27_195244']
old_part1_bootstrap = ['27_201503','27_205749','28_075635']
old_part1_MCbaseline = ['28_153349']
