#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter

eta_plots = ["28_013246"]

plot_this =eta_plots
rewards = []
colors = ['tab:blue','tab:orange','tab:green','tab:red','k','magenta','cyan']


for j, plot in enumerate(plot_this):
    rewards.append(np.load(f'data/rewards/r_{plot}.npy'))
    plt.plot(rewards,color=colors[j], label=plot)

plt.grid()
plt.legend()
plt.show()