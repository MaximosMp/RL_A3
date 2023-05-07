from REINFORCE_semi import reinforce
import time
import numpy as np


# game settings
n_episodes = 150
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
obs_type = 'vector'
boot = 'n_step'
baseline = True
seed = np.random.randint(100)
training = True

# give a list which weights you wanna run
# weights = ['26_175224']  # eta runs. len: 128, 168, 165 
# weights = ['06_174531', '06_183121', '06_191736']  # full run runs (we want 400, current:350)
# weights = ['07_025544']  # naked run
weights = ['06_103356','06_191148','06_202327']  # vectors

for weight in weights:
    P_weights = f'data/weights/w_P_{weight}.h5'
    V_weights = f'data/weights/w_V_{weight}.h5'
    stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))

    rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                        max_misses, max_steps, seed, n_step, speed, boot,
                        P_weights, V_weights, minibatch, eta, stamp, baseline, training)
