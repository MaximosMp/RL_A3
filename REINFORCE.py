import numpy as np
from catch import Catch
import tensorflow as tf
from collections import deque


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']

rng = np.random.default_rng(seed=None)

class Actor():
    def __init__(self, learning_rate: float = 0.0001, arch: int = 1, observation_type: str = "pixel", 
                 rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, boot: str = "MC", n_step: int = 1,
                 seed=42):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.speed = speed
        self.max_steps = max_steps
        self.max_misses = max_misses
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.boot = boot
        self.n_step = n_step

        self.gamma = 0.99
    
        # network parameters
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        # ENTROPY REGULARIZATION

        # gradient preparation
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        

        if (observation_type == 'pixel'):
            input_shape = (rows, columns, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)
        '''
        this is our current problem (layers.Dense) --> Flatten?
        Note: If the input to the layer has a rank greater than 2, then Dense computes 
        the dot product between the inputs and the kernel along the last axis of the inputs 
        and axis 0 of the kernel (using tf.tensordot). For example, if input has dimensions 
        (batch_size, d0, d1), then we create a kernel with shape (d1, units), and the kernel 
        operates along axis 2 of the input, on every sub-tensor of shape (1, 1, d1) (there are 
        batch_size * d0 such sub-tensors). The output in this case will have shape (batch_size, d0, units).
        '''

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

    def bootstrap(self, t, rewards, values = None):
        if self.boot == "MC":
            rewards = rewards[t:]
            return self.gamma**np.arange(0, len(rewards)) @ rewards
        elif self.boot == "n_step":
            rewards = rewards[t:t+self.n_step]
            return self.gamma**np.arange(0, self.n_step) @ rewards + self.gamma**self.n_step * values[t+self.n_step]

    def update_actor(self, state, action, Q):
        ''' so far I think the update needs to happen completely different than I did it here:
            - somehow we need to find gradient of log(pi(a,s))
            - we need to multiply this gradient by Q-value and (negative) learning rate (ascent, not descent)
        '''
        action_n = np.where(ACTION_EFFECTS==action)[0][0]
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = self.model(state, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            gain_value = self.gain_fn(state, action_n, Q)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(gain_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def gain_fn(self, state, action_n, Q):
        return Q * np.log(self.model.predict(state, verbose=0).reshape(3,)[action_n])

    def update_critic(self):
        pass


def reinforce():
    obs_type = "pixel"  # "vector"
    env = Catch(observation_type=obs_type)
    # state_shape = env.reset().shape

    learning_rate = 0.0001
    boot = "MC"

    n_episodes = 1000
    max_number_steps = 10
    if boot == "MC":
        n_step = max_number_steps
    elif boot == "n_step":
        n_step = 1  # or larger

    minibatch = 1

    actor = Actor(learning_rate, boot=boot, n_step=n_step)
    memory = deque(maxlen=max_number_steps)
    # Q_table = np.zeros((state_shape, len(ACTION_EFFECTS)))

    for ep in range(n_episodes):
        ep_reward = 0
        print(f'{ep}, reward: {ep_reward}')
        for m in range(minibatch):
            # gather trace
            state = env.reset()
            if obs_type == "pixel":
                state = state.reshape(1,actor.columns,actor.rows,2)
            elif obs_type == "vector":
                state = state.reshape(3,)  # maybe not needed
            print("state", state)

            for T in range(max_number_steps):
                if actor.boot == "MC":
                    print("SHAPE",state.shape)
                    action_p= actor.model.predict(state, verbose=0)
                    value = None
                elif actor.boot == "n_step":
                    action_p, value = actor.model.predict(state, verbose=0)
                action = rng.choice(ACTION_EFFECTS, p=action_p.reshape(3,))
                next_state, r, done = env.step(action)
                ep_reward += r
                memory.append((state,action,r,next_state,done, value))

                if done:
                    break
                state = next_state

            states, actions, rewards, next_states, dones, values = memory  # remove the ones we dont need
            for t in range(T):
                Q = actor.bootstrap(t,rewards, values)
                if actor.boot == 'MC':
                    actor.update_actor(states[t], actions[t],Q)


if __name__ == '__main__':
    reinforce()
