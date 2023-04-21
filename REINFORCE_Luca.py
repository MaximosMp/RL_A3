import numpy as np
from catch import Catch
import tensorflow as tf
from collections import deque
from Helper import time_it, print_it
import time

'''
    TO DO: 
        ask: 
            - whether gain_funct is ok (differentiability etc)
            - do we need -1 * Q in loss function?
            - REINFORCE is really online (inside t-loop)?
            - how many epochs/steps should we aim for? -> 200k-500k steps

        implement:
            - implement REINFORCE (online update) see comment in gain_fn
            - checkpoints (tensorboard)

        issue:
            - pixel state issue (vector is working)

        Experiments:
        1) REINFORCE, actor critic bootstrap, ac baseline, ac baseline + bootstrap (aren't the last two the same?)
            put them all in one plot, average over 5 runs, 1000 episodes
        2) vary size (5x5 converges faster): compare 5x5 vs 7x7, averaged over 5 runs and 1000-2000 traces
        3) Hyperparameter tuning (eta, learning rate)
        4) Other environment variations.
    '''


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']
seed = None
rng = np.random.default_rng(seed=seed)


def make_tensor(state, list: bool = False):
    '''in order to be used in model.predict() method'''
    s_tensor = tf.convert_to_tensor(state)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)


class Actor():
    # @time_it
    def __init__(self, learning_rate: float = 0.0001, arch: int = 1, observation_type: str = "pixel",
                 rows=7, columns=7, boot: str = "MC", n_step: int = 1, saved_weights: str = None,
                 seed=None, critic: bool = False, eta: float = 0.01, baseline: bool = False):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.boot = boot
        self.n_step = n_step
        self.critic = critic
        self.baseline = baseline
        self.eta = eta
        self.gamma = 0.99

        # network parameters
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        # IMPLEMENT ENTROPY REGULARIZATION

        # gradient preparation
        # self.loss_fn = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, )

        if (observation_type == 'pixel'):
            input_shape = (rows, columns, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)
        if arch == 1:
            input = tf.keras.layers.Input(shape=input_shape)
            dense = tf.keras.layers.Dense(
                32, activation=activ_func, kernel_initializer=init)(input)
            batchNorm = tf.keras.layers.BatchNormalization()(dense)
            dense = tf.keras.layers.Dense(
                32, activation=activ_func, kernel_initializer=init)(batchNorm)
            dropout = tf.keras.layers.Dropout(0.2)(dense)
            dense = tf.keras.layers.Flatten()(dropout)

        if critic:
            output_value = tf.keras.layers.Dense(1, activation='linear')(dense)
            self.model = tf.keras.models.Model(
                inputs=input, outputs=[output_value])
        else:
            output_actions = tf.keras.layers.Dense(
                3, activation='softmax')(dense)
            self.model = tf.keras.models.Model(
                inputs=input, outputs=[output_actions])

        if saved_weights:
            print('## Working with pre-trained weights ##')
            self.model.load_weights(saved_weights)
        self.model.summary()

    def bootstrap(self, t, rewards, values=None):
        if self.boot == "MC":
            rewards = rewards[t:]
            return self.gamma**np.arange(0, len(rewards)) @ rewards
        elif self.boot == "n_step":
            lim = min(t+self.n_step,len(rewards))
            rewards = rewards[t:lim]
            return self.gamma**np.arange(0, len(rewards)) @ rewards + self.gamma**self.n_step * values[lim-1]

    def gain_fn(self, prob_out=None, Q=None, actions=None):
        # if self.boot == 'MC':
        #     return -Q * tf.math.log(prob_out)
        actions = np.where(actions==-1,0, np.where(actions==0,1,2))  # rewrites [-1,0,1,-1,1,0,1] into [0,1,2,0,2,1,2]
        mask = tf.one_hot(actions, 3)  # mask = [[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,1]]
        prob_out = tf.reduce_max(mask * prob_out, axis=1) # prob_out = [0.2,0.3,0.5,0.2,0.5,0.3,0.3]

        Q_tensor = tf.constant(-1* np.ones(len(Q)) * Q,dtype=tf.float32) # Q_tensor = [-1,-1,-1,-1,-1,-1,-1]
        gain = Q_tensor * tf.math.log(prob_out) # shoud this be dot product?
        gain = tf.reduce_sum(gain)
        return gain

    def gain_fn_entropy(self,prob_out=None, Q=None, actions=None):
        actions = np.where(actions==-1,0, np.where(actions==0,1,2))  # rewrites [-1,0,1] into [0,1,2]
        mask = tf.one_hot(actions, 3)
        prob_out = tf.reduce_max(mask * prob_out, axis=1)

        gain = tf.tensordot(tf.constant(-1 * np.ones(len(Q)) * Q,dtype=tf.float32),
                            tf.math.log(prob_out), 1)
        gain -= self.eta* tf.tensordot(prob_out, tf.math.log(prob_out),1) # -sum_i ( pi * log(pi) )
        return gain

    def update_weights(self, states, actions, Q_values, values=None):
        '''got code structure from https://keras.io/guides/writing_a_training_loop_from_scratch/'''
        states = tf.convert_to_tensor(states)
        
        with tf.GradientTape() as tape:
            if self.critic:
                values = self.model(states)
                gain_value = tf.losses.mean_squared_error(Q_values,values)
            else:
                probs_out = self.model(states)
                gain_value = self.gain_fn(prob_out=probs_out, Q=Q_values, actions=actions)
                print("PROBS", probs_out[-10:])
            
        grads = tape.gradient(gain_value,
                              self.model.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        # norm_grads = [tf.norm(grad) for grad in grads]
        return grads

    def reshape_state(self, state):
        if self.observation_type == "pixel":
            return state.reshape(1, self.columns, self.rows, 2)
        elif self.observation_type == "vector":
            return state.reshape(1, 3)

    def unshape_state(self, state):
        '''same is done by squeeze'''
        if self.observation_type == "pixel":
            return state.reshape(self.columns, self.rows, 2)
        elif self.observation_type == "vector":
            return state.reshape(3)


@time_it
def reinforce(n_episodes: int = 50, learning_rate: float = 0.001, rows: int = 7, columns: int = 7, 
              obs_type: str = "pixel", max_misses: int = 10, max_steps: int = 250, seed: int = None, 
              n_step: int = 5, speed: float = 1.0, boot: str = "MC", weights: str = None, 
              minibatch: int = 1, eta: float = 0.01, stamp: str = None, baseline: bool = False):
    if boot == "MC":
        baseline = False

    # IMPLEMENT (TENSORBOARD) CALLBACKS FOR ANALYZATION, long book 315

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=obs_type, seed=None)

    # NON-training average is around -8.4. So we are only learning when we're significantly higher (let's say < -7.0)

    if boot == "MC":
        n_step = max_steps

    all_rewards = []
    actor = Actor(learning_rate, boot=boot, n_step=n_step,
                  observation_type=obs_type, saved_weights=weights, 
                  seed=seed, eta=eta, baseline=baseline)

    if boot == 'n_step':
        critic = Actor(learning_rate, boot=boot, n_step=n_step,
                       observation_type=obs_type, saved_weights=weights, seed=seed, 
                       critic=True, eta=eta)
    memory = deque(maxlen=max_steps)
    count = 0

    for ep in range(n_episodes):
        print()
        for m in range(minibatch):
            ep_reward = 0
            memory.clear()
            state = actor.reshape_state(env.reset())
            # generate full trace
            for T in range(max_steps):
                action_p = actor.model.predict(state, verbose=0)

                if actor.boot == "n_step":
                    value = critic.model.predict(state, verbose=0)
                    value = tf.squeeze(value)
                elif actor.boot == "MC":
                    value = None

                action = np.random.choice(
                    ACTION_EFFECTS, p=action_p.reshape(3,))

                next_state, r, done = env.step(action)
                count += 1
                next_state = actor.reshape_state(next_state)
                ep_reward += r

                # take out the extra "1" dimensions
                memory.append((tf.squeeze(state),
                              action, r, next_state, done, value))

                if done:
                    break
                state = next_state

            # trace is finished
            print(f'{ep}, step {count}, reward: {ep_reward}')
            all_rewards.append(ep_reward)

        # extract data from memory (to do: delete unused variables)
        memory = [memory[index] for index in range(len(memory))]
        states, actions, rewards, next_states, dones, values = [np.array([experience[field_index]
                                                                          for experience in memory])
                                                                for field_index in range(6)]


        Q_values = [actor.bootstrap(t,rewards,values) for t in range(T+1)]

        left = np.sum(np.where(actions==-1,1,0))
        idle = np.sum(np.where(actions==0,1,0))
        right = np.sum(np.where(actions==1,1,0))
        # print(f"left {left}, idle {idle}, right {right}")
        if actor.boot == 'n_step':
            # in case V network is updated separately!
            critic.update_weights(states, actions, Q_values)
        if actor.baseline:
            A_values = [Q_values[i]-values[i] for i in range(T+1)]
            actor.update_weights(states, actions, A_values)
        else:
            actor.update_weights(states, actions, Q_values)


        # manual grad clipping
        # is_threshold = [tf.norm(grad) < 0.0000001 for grad in grads]

        # if (True in is_threshold):
        #     print("BREAK!")
        #     break

<<<<<<< HEAD:REINFORCE_semi.py
=======
        if actor.boot == 'n_step' and Training:
            # in case V network is updated separately!
            critic.update_weights(states, actions, rewards, Q_values)
>>>>>>> c61122fa2a441424dda05d6ec91860da7e9d3148:REINFORCE_Luca.py

        if ep % 10 == 0 and ep > 0:
            np.save(f'data/rewards/tmp_reward', all_rewards)

    actor.model.save_weights(f'data/weights/w_{stamp}.h5')
    actor.model.save_weights(f'data/weights/latest_weights.h5')
    np.save(f'data/rewards/r_{stamp}', all_rewards)
    if seed:
        print(f'ran with seed {seed}!')
    if weights:
        print(f'ran on pre-trained weights')
    return all_rewards


if __name__ == '__main__':

    # game settings
<<<<<<< HEAD:REINFORCE_semi.py
    n_episodes = 600
    learning_rate = 0.001
    rows = 5
    columns = 5
    obs_type = "vector"  # "vector" or "pixel"
=======
    n_episodes = 200
    learning_rate = 0.0001
    rows = 7
    columns = 7
    obs_type = "pixel"  # "vector" or "pixel"
>>>>>>> c61122fa2a441424dda05d6ec91860da7e9d3148:REINFORCE_Luca.py
    max_misses = 10
    max_steps = 250
    seed = None  # if you change this, change also above! (at very beginning)
    n_step = 8
    speed = 1.0
    boot = "MC"  # "n_step" or "MC"
    minibatch = 1
    weights = None
    baseline = False
    # weights = 'data/weights/w_18_184522.h5'

    ### hyperparameters to tune
    # etas = [0.0001, 0.001, 0.01, 0.1]
    etas = [0.1]
    # learning_rates = [0.1, 0.01, 0.001, 0.0001]
    learning_rates = [0.001]
    for learning_rate in learning_rates:
        for eta in etas:
            start = time.time()
            stamp = time.strftime("%d_%H%M%S", time.gmtime(start))
            rewards = reinforce(n_episodes, learning_rate, rows, columns, obs_type,
                                max_misses, max_steps, seed, n_step, speed, boot, 
                                weights, minibatch, eta, stamp, baseline)
            with open("data/documentation.txt", 'a') as f:
                # export comand line output for later investigation
                f.write(
                    f'\n\nStamp: {stamp} ... Episodes: {n_episodes}, Learning: {learning_rate}, Seed: {seed}, '
                    + f'Eta: {eta}, Avg reward: {np.mean(rewards)}\n')
