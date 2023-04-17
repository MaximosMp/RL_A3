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
            rewards.append(np.load(f'data/rewards/{arg}'))
            titles.append(arg)
        except:
            continue

    
    colors = ['tab:blue','tab:orange','tab:green','tab:red','k','magenta','cyan']
    for j, reward in enumerate(rewards):
        avg = round(np.mean(reward),3)
        median = np.median(reward)
    
        print(f"Average performance {args[j]}: {avg}")
        # if "mean" in args:
        #     reward = np.mean(reward,axis=0)
        # plt.title(titles[j])
        # plt.plot(reward,alpha=0.2,color=colors[j],label=f"Raw data {j}")
        plt.plot(savgol_filter(reward,10,polyorder=1),color=colors[j],label=f"Convolution {j}")  # end always goes down due to conv
        plt.axhline(avg, color=colors[j],linewidth=1,label= "Average", linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plt.legend()
    plt.grid()
    #plt.savefig(f"data/rewards_{j}.pdf")
    plt.show()
