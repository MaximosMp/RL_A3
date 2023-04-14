import numpy as np
from catch import Catch
import tensorflow as tf

# create the REINFORCE algorithm
# def REINFORCE():

env = Catch()


ACTION_EFFECTS = (-1, 0, 1)  # left, idle right.
OBSERVATION_TYPES = ['pixel', 'vector']


class Actor():
    def __init__(self, learning_rate: float = 0.0001, arch: int = 1, observation_type: str = "pixel", rows=7, columns=7, speed=1.0, max_steps=250, max_misses=10, seed=42):
        self.seed = seed
        self.rows = rows
        self.columns = columns
        self.speed = speed
        self.max_steps = max_steps
        self.max_misses = max_misses
        self.observation_type = observation_type
        self.learning_rate = learning_rate
        self.observation_type = observation_type
        self.gamma = 0.99
        activ_func = "relu"
        init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        self.loss_fn = tf.keras.losses.mean_squared_error

        if (observation_type == 'pixel'):
            input_shape = (rows, columns, 2)
        elif (observation_type == 'vector'):
            input_shape = (3,)

        input = tf.keras.layers.Input(shape=input_shape)
        dense = tf.keras.layers.Dense(
            64, activation=activ_func, kernel_initializer=init)(input)
        dense = tf.keras.layers.Dense(
            64, activation=activ_func, kernel_initializer=init)(dense)

        output_actions = tf.keras.layers.Dense(3, activation='softmax')(dense)
        output_value = tf.keras.layers.Dense(1, activation='linear')(dense)

        self.model = tf.keras.models.Model(inputs=[input], outputs=[
            output_actions, output_value])

        self.model.summary()


# def update_actor(self, Q_table, V_table=None):
#     pass


# class REINFORCE():
def reinforce():
    learning_rate = 0.0001
    n_step = 1
    number_of_episodes = 1000
    max_number_steps = 10
    actor = Actor(learning_rate)

    for t in range(max_number_steps):
        state = env.reset()
        print('state', state.shape)
        action, value = actor.model.predict(state.reshape(1,7,7,2))
        print(action)
        next_state, r, done = env.step(action)


if __name__ == '__main__':
    reinforce()
