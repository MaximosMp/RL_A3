#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot one or more saved reward files against each other.

usage: python plot.py data/rewards/r_<stamp>.npy [...]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

args = sys.argv[1:]

if __name__ == "__main__":
    rewards = []
    titles = []
    for arg in args:
        try:
            rewards.append(np.load(arg))
            titles.append(arg)
        except (OSError, ValueError) as err:
            print(f"skipping {arg}: {err}", file=sys.stderr)

    if not rewards:
        raise SystemExit(__doc__.split("usage: ", 1)[1].rstrip())

    
    colors = ['tab:blue','tab:orange','tab:green','tab:red','k','magenta','cyan']
    for j, reward in enumerate(rewards):
        avg = np.mean(reward)
        color = colors[j % len(colors)]
        print(f"Average performance {titles[j]}: {avg:.3f} after {len(reward)} episodes. Avg last 50: {np.mean(reward[-50:]):.3f}\n")
        plt.title(titles[j])
        plt.plot(reward,alpha=1.,color=color)
        # the smoothed tail always dips, an artefact of the convolution window
        plt.plot(savgol_filter(reward,10,polyorder=1),color=color,label=f" {titles[j][-15:]}",alpha=0.1)
        plt.axhline(avg, color=color,linewidth=1,label= "Average", linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.xlim(0,100)
    plt.legend()
    plt.grid()
    plt.savefig("data/tmp_plot.pdf")
    plt.show()
