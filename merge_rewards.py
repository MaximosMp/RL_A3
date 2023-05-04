import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


'''
in case the run was interrupted but we want a single combined rewards file
merge first n-1 files (input as stamp) together to one reward file and save it at n-th path:

python merge_rewards.py 03_140000 03_100155 custom_name
'''

args = sys.argv[1:]

rewards = np.array([])

for plot in args[:-1]:
    rewards = np.append(rewards,np.load(f'data/rewards/{plot}.npy'))

np.save(f'data/rewards/r_{args[-1]}',rewards)
plt.plot(savgol_filter(rewards,10,1))
plt.show()
