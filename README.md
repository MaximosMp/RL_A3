# Deep Reinforcement Learning - Assignment 3 - Group 13
Our main classes and functions are in the `policy.py` file and depending on the experiment you want to recreate, use the following flags.

The base hyperparameters are listed below and can be changed in the `policy.py` file.
```
    n_repetitions = 1
    n_episodes = 300
    learning_rate = 0.01
    rows = 7
    columns = 7
    obs_type = "pixel"  # "vector" or "pixel"
    max_misses = 10
    max_steps = 250
    n_step = 5
    speed = 1.
    eta = 0.001
    minibatch = 4
```

## Monte Carlo, N-step, Baseline, PPO
### Run
Running the file without any flags, will execute the **n-step baseline** model for 1 repetition in 300 episodes
```
python policy.py
```

#### Add optional flags for:  
Running the **Monte Carlo** model without **baseline**

```
python policy.py --mc
```

Running the **Monte Carlo** model with **baseline**

```
python policy.py --mc --baseline
```

Running the **n-step** model without **baseline**

```
python policy.py --n_step
```

Running the **n-step** model with **baseline**

```
python policy.py --n_step --baseline
```

Running the **PPO** model

```
python policy.py --ppo
```

Please do not use other combinations than those mentioned above. We did not make it fully functional and self-consistent.

