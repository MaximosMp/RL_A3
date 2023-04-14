import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from collections import deque  # for experience replay
from Helper import make_tensor, e_greedy, softmax, linear_anneal, convolute
import matplotlib.pyplot as plt
import sys


class Q_Network():
    '''General deep Q Network'''

    def __init__(self, learning_rate: float = 0.0001, optimizer: str = 'adam', architecture: int = 4,
                 path_to_weights: str = None, batch_size: int = 32, target_active: bool = True, is_target: bool = False, double_dqn: bool = False):
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.gamma = 0.99
        self.update_count = 0
        self.target_active = target_active
        activ = "elu"
        self.seed = 42
        init = tf.keras.initializers.HeNormal(seed=self.seed)
        self.loss_fn = keras.losses.mean_squared_error
        self.path_to_weights = path_to_weights
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.is_target = is_target
        self.double_dqn = double_dqn

        if architecture == 1:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation=activ, input_shape=(
                    4,), kernel_initializer=init),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 2:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    30, activation=activ, kernel_initializer=init, input_shape=(4,)),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 3:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=activ, input_shape=(
                    4,), kernel_initializer=init),
                tf.keras.layers.Dense(32, activation=activ),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 4:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=activ, input_shape=(
                    4,), kernel_initializer=init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation=activ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2)
            ])
        elif architecture == 'Dueling':
            K = keras.backend

            input_states = keras.layers.Input(shape=[4])
            hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
            hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)

            # This creates a fully connected layer with a single neuron and no activation function, which will output the estimated state value for the input state.
            state_values = keras.layers.Dense(1)(hidden2)

            # This creates another fully connected layer with n_outputs neurons and no activation function, which will output the raw advantage estimates for each action.
            raw_advantages = keras.layers.Dense(2)(hidden2)

            # This calculates the advantages by subtracting the maximum advantage estimate from each estimate, which helps to stabilize the learning process.
            advantages = raw_advantages - \
                K.max(raw_advantages, axis=1, keepdims=True)

            # This combines the state values and advantages to compute the Q-values for each action.
            Q_values = state_values + advantages

            # now the model has two outputs, one for the state values and one for the advantages.
            self.model = keras.Model(inputs=[input_states], outputs=[Q_values])

        if path_to_weights:
            print("using pre-trained weights")
            self.learning = False
            full_path_to_weights = path_to_weights
            self.model.load_weights(full_path_to_weights)
        else:
            self.learning = True
            self.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            self.model.compile(self.optimizer, self.loss_fn)
        if not self.is_target:
            self.model.summary()

    def update(self, states, actions, rewards, next_states, dones, target):
        self.update_count += 1

        # self.model.fit(states, target_Q_values, batch_size=self.batch_size, verbose=0)

        if self.double_dqn:

            next_Q_values = self.model.predict(next_states,verbose=0)
            best_next_actions = np.argmax(next_Q_values, axis=1)
            next_mask = tf.one_hot(best_next_actions, 2).numpy()
            next_best_Q_values = (target.model.predict(
                next_states,verbose=0) * next_mask).sum(axis=1)
            target_Q_values = (rewards +
                               (1 - dones) * self.gamma * next_best_Q_values)
            mask = tf.one_hot(actions, 2)
        else:
            target_output = target.model.predict(next_states, verbose=0)
            max_next_Q_values = np.max(target_output, axis=1)
            target_Q_values = (rewards +
                               (1 - dones) * self.gamma * max_next_Q_values)

        mask = tf.one_hot(actions, 2)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(
                all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))


class DQN_Agent():
    '''Class for the learning and playing agent'''

    def __init__(self, batch_size: int = 32, epsilon: float = 0.02, temp: float = 1, max_buffer: int = int(1e3),
                 policy: str = 'epsilon', replay_active: bool = True):
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.temp_initial = temp
        self.temp = temp
        self.policy = policy
        self.batch_size = batch_size
        self.replay_active = replay_active
        self.buffer = deque(maxlen=max_buffer)

    def draw_action(self, state, network: Q_Network):
        '''
        input: 
        - s: (4dim) state of environment
        returns:
        - int 0 or 1, action according to QNet and policy
        no learning is happening here '''

        if self.policy == 'epsilon':
            return e_greedy(self.epsilon, state, network)
        elif self.policy == 'softmax':
            return softmax(self.temp, state, network)
        elif self.policy == 'greedy':
            s_tensor = make_tensor(state, False)
            Q_vals = network.model.predict(
                s_tensor, verbose=0)  # outputs two Q-values
            return np.argmax(Q_vals)

    def draw_sample(self):
        ''' append newly obtained environment state to memory'''
        if self.replay_active:
            indices = np.random.randint(len(self.buffer), size=self.batch_size)
        else:
            indices = np.arange(-self.batch_size, 0, 1)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def buffer_update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


def learning(eps, learning_rate, batch_size, architecture, target_update_freq, replay_buffer_size, policy: str = 'epsilon',
             epsilon: float = 0.02, temp: float = 1., path_to_weights: str = None, replay_active: bool = True, target_active: bool = True,
             double_dqn: bool = False, render: bool = False):
    np.random.seed(42)
    start = time.time()
    stamp = time.strftime("%d_%H%M%S", time.gmtime(start))
    # params:{learning_rate, batch_size, architecture, target_update_freq, replay_buffer_size}
    outp = f"---Starting Test {stamp}---\n"
    print(outp)

    network = Q_Network(learning_rate, 'adam', architecture,
                        path_to_weights=path_to_weights, target_active=target_active, double_dqn=double_dqn)
    target = Q_Network(is_target=True)
    target.model = tf.keras.models.clone_model(network.model)
    target.model.set_weights(network.model.get_weights())

    if not network.learning:  # when path to weights is given, we exploit fully
        policy = 'greedy'

    agent = DQN_Agent(batch_size, epsilon=epsilon, temp=temp,
                      max_buffer=replay_buffer_size, policy=policy, replay_active=replay_active)

    # start the environment
    if render:
        env = gym.make("CartPole-v1", render_mode='human')
    else:
        env = gym.make("CartPole-v1")

    ep_rewards = []
    max_mean = 60
    budget = 0

    for episode in range(eps):
        state, info = env.reset()

        # agent.epsilon = max(1 - episode / 500, 0.01)
        agent.epsilon = linear_anneal(
            episode, eps, agent.epsilon_initial, 0.01, 0.5)
        agent.temp = linear_anneal(episode, eps, agent.temp_initial, 0.01, 0.5)
        # epsilon = max(1 - np.mean(ep_rewards)/200, 0.01)  # idea: couple annealing epsilon not to ep count but reward?
        cumulative_reward = 0

        # episode starts
        while True:
            budget += 1
            action = agent.draw_action(state, network)
            next_state, reward, term, trunk, info = env.step(action=action)
            done = trunk or term

            agent.buffer_update(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward

            if done:
                break
        # episode is over
        # model training when no existing weight path is given
        if episode > 50 and network.learning:
            if episode == 51:
                prnt = "We're now learning..."
                outp += "\n" + prnt
                print(prnt)

            # do training
            states, actions, rewards, next_states, dones = agent.draw_sample()
            network.update(states, actions, rewards,
                           next_states, dones, target)

            # update target (every episode when target network is False)
            if not network.target_active:
                target.model.set_weights(network.model.get_weights())
            elif network.update_count % target_update_freq == 0:
                target.model.set_weights(network.model.get_weights())

        elif not network.learning:
            if episode == 0:
                prnt = "Not learning, but playing..."
                outp += "\n" + prnt
                print(prnt)

        ep_rewards += [cumulative_reward]

        try:
            mean = round(np.mean(ep_rewards[-50:]), 3)
        except:
            mean = round(np.mean(ep_rewards), 3)

        if episode % 50 == 0 and episode > 50:
            prnt = f"Average of last 50: {mean}, Ep: {episode}"
            outp += prnt
            print(prnt)

        if episode % 10 == 0:
            np.save(f"rew_{stamp}.npy", np.array(ep_rewards))

        if mean > max_mean and network.learning:
            max_mean = mean
            prnt = f"Saving weights at mean reward: {mean}..."
            outp += "\n"+prnt
            network.model.save_weights(
                f"w_{stamp}.h5", overwrite=True)

    # all episodes are done
        # save the mean/median after episodes are finished
    rew_mean = round(np.mean(ep_rewards[50:]), 3)
    rew_median = np.median(ep_rewards[50:], axis=0)

    ticks = round((time.time()-start)/60, 2)
    prnt = f"It took {ticks}mins."
    outp += "\n" + prnt + "\n"
    print(prnt)

    prnt = f"\nMean reward:{rew_mean}, median: {rew_median}\nTime: {ticks}mins\n\n"
    print(prnt)
    outp += prnt

    with open("documentation.txt", 'a') as f:
        # export comand line output for later investigation
        f.write("\n"+outp)
    env.close()
    return ep_rewards


if __name__ == "__main__":
    # this file can be run with: python dqn.py --target_active --experience_replay

    args = sys.argv[1:]
    target_active, replay_active = False, False
    training = True
    double_dqn = False
    dueling = False
    try:
        for arg in args:
            if arg == "--target_active":
                target_active = True
                print("\nTarget network active...\n")
                continue
            elif arg == "--experience_replay":
                replay_active = True
                print("\nExperience replay active...\n")
                continue
            elif arg == "--no_training":
                training = False
                break
            elif arg == "--dueling":
                dueling = True
                break
            elif arg == "--double":
                double_dqn = True
                break
    except:
        pass

    eps = 500
    n_runs = 1

    all_rewards = np.empty([n_runs, eps])

    policy = "epsilon"
    path_to_weights = None

    learning_rate, batch_size, arch, target_update_freq, replay_buffer_size = (
        0.0001, 32, 4, 10, 5000)

    if not training:
        path_to_weights = "good_w_01_112419.h5"
        arch = 1
    if dueling:
        arch = "Dueling"

    temp = 1
    epsilon = 1.0

    for run in range(n_runs):
        rewards = learning(eps, learning_rate, batch_size, architecture=arch, target_update_freq=target_update_freq,
                           replay_buffer_size=replay_buffer_size, policy=policy, epsilon=epsilon,
                           path_to_weights=path_to_weights, temp=temp, replay_active=replay_active, target_active=target_active, double_dqn=double_dqn)
        all_rewards[run] = rewards

    if run > 0:
        plt.plot(np.mean(all_rewards, axis=0), alpha=0.2,
                 color='tab:blue', label='Raw data')
        plt.plot(convolute(np.mean(all_rewards, axis=0)),
                 color='tab:blue', label='Convoluted')
    else:
        plt.plot(rewards, alpha=0.2, color='tab:b', label='Raw data')
        plt.plot(convolute(rewards), color='tab:b', label='Convoluted')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    stamp = time.strftime("%d_%H%M%S", time.gmtime(time.time()))

    if target_active or replay_active:
        plt.title(f"Learning with TN:{target_active} ER:{replay_active}")
        plt.savefig(f"exp_TN:{target_active}_ER:{replay_active}_{stamp}.pdf")
        np.save(
            f"rewards_TN({target_active})_ER({replay_active})_{stamp}", all_rewards)
    elif not training:
        plt.title(f"Performance on pre-trained weights")
        plt.savefig(f"pre-trained_{stamp}.pdf")
        np.save(f"pre-trained_{stamp}", all_rewards)
    elif double_dqn:
        plt.title(f"Double Deep Q Network")
        plt.savefig(f"double_dqn_{stamp}.pdf")
        np.save(f"rewards_double_dqn_{stamp}", all_rewards)
    elif dueling:
        plt.title(f"Dueling Deep Q Network")
        plt.savefig(f"dueling_dqn_{stamp}.pdf")
        np.save(f"rewards_dueling_dqn_{stamp}", all_rewards)
