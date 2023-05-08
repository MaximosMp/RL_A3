#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter


args = sys.argv[1:]

run = 1
count = 0


if __name__ == "__main__":
## file to plot 
    rewards = []
    titles = []
    for arg in args:
        try:
            rewards.append(np.load(f'{arg}'))
            titles.append(arg)
        except:
            continue

    
    colors = ['tab:blue','tab:orange','tab:green','tab:red','k','magenta','cyan']
    for j, reward in enumerate(rewards):
        avg = np.mean(reward)
        median = np.median(reward)
        print(f"Average performance {args[j]}: {avg:.3f} after {len(reward)} episodes. Avg last 50: {np.mean(reward[-50:]):.3f}\n")
        # if "mean" in args:
        #     reward = np.mean(reward,axis=0)
        plt.title(titles[j])
        plt.plot(reward,alpha=1.,color=colors[j])
        plt.plot(savgol_filter(reward,10,polyorder=1),color=colors[j],label=f" {titles[j][-15:]}",alpha=0.1)  # end always goes down due to conv
        plt.axhline(avg, color=colors[j],linewidth=1,label= "Average", linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(f"data/tmp_plot.pdf")
    plt.show()
