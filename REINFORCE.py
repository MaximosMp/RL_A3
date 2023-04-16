import numpy as np
from catch import Catch
import tensorflow as tf
from collections import deque
from Helper import *


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']

rng = np.random.default_rng(seed=None)

def make_tensor(state, list: bool = False):
    '''in order to be used in model.predict() method'''
    s_tensor = tf.convert_to_tensor(state)
    if list:
        return s_tensor
    return tf.expand_dims(s_tensor, 0)

class Actor():
    @time_it
    def __init__(self, learning_rate: float = 0.0001, arch: int = 1, observation_type: str = "pixel", 
                 rows=7, columns=7, boot: str = "MC", n_step: int = 1,
                 seed=None):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.boot = boot
        self.n_step = n_step

        self.gamma = 0.99
    
        # network parameters
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        # IMPLEMENT ENTROPY REGULARIZATION

        # gradient preparation
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        

        if (observation_type == 'pixel'):
            input_shape = (rows, columns, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)
        if arch == 1:
            input = tf.keras.layers.Input(shape=input_shape)
            dense = tf.keras.layers.Dense(
                64, activation=activ_func, kernel_initializer=init)(input)
            dense = tf.keras.layers.Dense(
                64, activation=activ_func, kernel_initializer=init)(dense)
            dense = tf.keras.layers.Flatten()(dense)

            output_actions = tf.keras.layers.Dense(3, activation='softmax')(dense)

        if boot == "MC":
            self.model = tf.keras.models.Model(inputs=input, outputs=[
            output_actions])

        elif boot == "n_step":
            output_value = tf.keras.layers.Dense(1, activation='linear')(dense)
            self.model = tf.keras.models.Model(inputs=input, outputs=[
                output_actions, output_value])

        self.model.summary()

    # @print_it
    def bootstrap(self, t, rewards, values = None):
        if self.boot == "MC":
            rewards = rewards[t:]
            return self.gamma**np.arange(0, len(rewards)) @ rewards
        elif self.boot == "n_step":
            rewards = rewards[t:t+self.n_step]
            return self.gamma**np.arange(0, self.n_step) @ rewards + self.gamma**self.n_step * values[t+self.n_step]

    def gain_fn(self, prob_out, Q):
        gain = -Q * tf.math.log(prob_out)  # * -1 due to maximising instead of minimizing?
        return gain

    def update_actor(self, state, action, Q):
        action_n = np.where(ACTION_EFFECTS==action)[0][0]
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            if self.observation_type == "pixel":
                state = make_tensor(state.reshape(self.columns,self.rows,2))
            elif self.observation_type == "vector":
                state = make_tensor(state.reshape(3,))
            if self.boot == "MC":
                probs_out = self.model(state, training=True)
                value_out = None
            elif self.boot == "n_step":
                probs_out, value_out = self.model(state, training=True)
            # Compute the loss value for this minibatch.
            gain_value = self.gain_fn(tf.reshape(probs_out, [3])[action_n], Q)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(gain_value, self.model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def update_critic(self):
        pass

    def reshape_state(self, state):
        if self.observation_type == "pixel":
            return state.reshape(1,self.columns,self.rows,2)
        elif self.observation_type == "vector":
            return state.reshape(1,3)

@time_it
def reinforce():
    # game settings
    rows = 7
    columns = 7
    obs_type = "vector"  # "vector"
    max_misses = 10
    max_steps = 250
    seed = None
    speed = 1.0

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps, max_misses=max_misses, observation_type=obs_type)

    learning_rate = 0.001
    boot = "MC"

    n_episodes = 100
    if boot == "MC":
        n_step = max_steps
    elif boot == "n_step":
        n_step = 1  # or larger

    minibatch = 1
    all_rewards = []
    actor = Actor(learning_rate, boot=boot, n_step=n_step, observation_type=obs_type)
    memory = deque(maxlen=max_steps)

    for ep in range(n_episodes):
        ep_reward = 0
        memory.clear()
        for m in range(minibatch):
            '''
            PROBLEM: For some reason after some episodes the output of the network
            is "nan" (the probabilities are non existent). Next step: figure this out
            my suspect: when ball drops in X=0,1,6 we have a problem
            '''
            state = actor.reshape_state(env.reset())
            print(f"{ep} starting", state)
            # generate full trace
            for T in range(max_steps):
                if actor.boot == "MC":
                    action_p= actor.model.predict(state, verbose=0)
                    value = None
                elif actor.boot == "n_step":
                    action_p, value = actor.model.predict(state, verbose=0)
                try:
                    action = rng.choice(ACTION_EFFECTS, p=action_p.reshape(3,))
                    # print(f"good state: {state}")
                    # print(f"good probabilities:",action_p)
                except:
                    print(f"step {T}, faulty state: {state}")
                    print(f"faulty probabilities:",action_p)
                    action = rng.choice(ACTION_EFFECTS, p=[0.33,0.33,0.34])

                next_state, r, done = env.step(action)
                next_state = actor.reshape_state(next_state)
                ep_reward += r
                memory.append((state,action,r,next_state,done,value))
                # each sarsa combination is in one row -> too many entries. fix so that each part gets returned separately

                if done:
                    print(f'Terminated after {T} steps (ep: {ep})')
                    break
                state = next_state

            # trace is finished
            print(f'{ep}, reward: {ep_reward}')
            all_rewards.append(ep_reward)

            # extract data from memory (to do: delete unused variables)
            memory = [memory[index] for index in range(len(memory))]
            states, actions, rewards, next_states, dones, values = [np.array([experience[field_index] 
                                                                      for experience in memory]) 
                                                                      for field_index in range(6)]
            # update loop
            for t in range(T):
                Q = actor.bootstrap(t,rewards, values)
                if actor.boot == 'MC':
                    actor.update_actor(states[t], actions[t],Q)
                elif actor.boot == 'n_step':
                    print('insert actor critic bootstrap (and later baseline subtr.)')
                    break

    return all_rewards

if __name__ == '__main__':
    print(reinforce())
