import os
import time

import numpy as np
import tensorflow.compat.v1 as tf

from atari_wrappers import make_atari, wrap_deepmind


## Global TensorFlow Configuration
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

# Instruct TensorFlow to use only the first GPU of the system.
gpu_devices = gpu_devices[2:3]
tf.config.experimental.set_visible_devices(gpu_devices, 'GPU')

# Avoid allocating all GPU memory upfront.
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Instruct the Linux kernel to preferably kill this process instead of its
# ancestors in an OOM situation. This prevents cases in which the SSH server is
# killed, blocking any further access to the machine.
# FIXME
with open('/proc/self/oom_score_adj', 'w') as f:
    f.write('1000\n')


class ReplayBuffer(object):
    def __init__(self):
        self.counter = 0
        self.capacity = 10000
        self.buffer = np.empty([self.capacity, 5], dtype=object)

    def append(self, data):
        """

        :param data:
        :return:
        """
        self.buffer[self.counter] = data
        self.counter = np.mod(self.counter + 1, self.capacity)

    def sample(self, B):
        """
        :return:
        """
        idx = np.random.choice(self.capacity, B)
        return self.buffer[idx]


def check_dir(directory):
    """
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def wrap_atari_deepmind(environment_name, clip_rewards):
    """
    Receive the environment name and the boolean variable for clipping the reward, that should only be clipped
    during training, not evaluation.
    Use make atari to create the environment by name, and wrap the resulting environment with the wrap deepmind
    function. You will use episode_life=True, frame_stack=True, scale=True, and clip rewards as specified from the
    input to your new function.

    :param environment_name: the environment name
    :param clip_rewards: boolean variable that indicates whether the reward should be clipped
    :return: the wrapped environment
    """

    env = make_atari(environment_name)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=clip_rewards, frame_stack=True, scale=True)

    return env


def create_conv_layer(filter_size, stride, input_size, output_size, input):
    """

    :param filter_size: size of the filter
    :param stride: stride
    :param input_size: input size
    :param output_size: number of neurons
    :param input: input of the layer
    :return A_conv
    """
    with tf.variable_scope(None, default_name="conv"):
        # Initialize the weights with tf.variance_scaling_initializer
        W_init = tf.variance_scaling_initializer()
        W_conv = tf.Variable(W_init([filter_size, filter_size, input_size, output_size]))

        # Initialize the biases with tf.zeros initializer
        b_init = tf.zeros_initializer()
        b_conv = tf.Variable(b_init(output_size))

        A_conv = tf.nn.conv2d(input, W_conv, strides=[1, stride, stride, 1], padding='SAME') + b_conv

        # Apply activation function
        A_conv = tf.nn.relu(A_conv)

    return A_conv


def create_fc_layer(input_size, output_size, input, activation):
    """

    :param input_size: input size
    :param output_size: number of neurons
    :param input: input of the layer
    :param activation: boolean parameter that if is True the rectified linear activations is applied
    :return A_fc
    """
    with tf.variable_scope(None, default_name="fc"):
        W_fc = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        b_fc = tf.Variable(tf.zeros(shape=(output_size,)))
        A_fc = tf.matmul(input, W_fc) + b_fc

        if activation:
            A_fc = tf.nn.relu(A_fc)

    return A_fc


def conv_net(X, k):
    """

    :param X: input
    :param k: number of neurons for the fully connected layer
    :return A_fc2: output
    """
    input_shape = X.shape.as_list()[-1]

    # Convolutional layer 1: 32 filters, 8 × 8.
    A_conv1 = create_conv_layer(8, 4, input_shape, 32, X)

    # Convolutional layer 2: 64 filters, 4 × 4.
    A_conv2 = create_conv_layer(4, 2, 32, 64, A_conv1)

    # Convolutional layer 3: 64 filters, 3 × 3.
    A_conv3 = create_conv_layer(3, 1, 64, 64, A_conv2)

    A_conv3_shapes = A_conv3.shape.as_list()
    A_conv3_units = np.product(A_conv3_shapes[1:])
    A_conv3_flat = tf.reshape(A_conv3, [-1, A_conv3_units])

    # Fully connected layer 1: 512 units.
    A_fc1 = create_fc_layer(A_conv3_units, 512, A_conv3_flat, True)

    # Fully connected layer 2: k units.
    A_fc2 = create_fc_layer(512, k, A_fc1, False)

    return A_fc2


def net_param(model, k):
    """

    :param model: current model
    :param learning_rate: learning rate of the model
    :param neurons_fc: number of units for the fully connected layer
    :return X, Z, argmax_Z, loss, accuracy, train
    """
    with tf.variable_scope("model_{}".format(model)) as scope:
        X = tf.placeholder(tf.float32, [None, 84, 84, 4], name='X')

        Z = conv_net(X, k)
        if model == 'online':
            out = tf.argmax(Z, axis=1)
        else:
            out = tf.reduce_max(Z, axis=1, keepdims=True)

    return X, Z, out, scope


def create_loss(decay, learning_rate, gamma, Z_o, A, R, max_Z_t, Omega):
    """

    :param decay:
    :param learning_rate:
    :param gamma:
    :param Z_o:
    :param a:
    :param r:
    :param max_Z_t:
    :param omega1:
    :return loss, train:
    """
    # if Ω′ = 1 -> y = r
    # elif Ω′ = 0 -> y = r + γ * max_a′ Q(s′, a′; θ′)
    y = tf.where(Omega, R, R + (gamma * max_Z_t))

    # Let L(θ) = 􏰀sum􏰀_{(s, a, r, s′, Ω′) ∈ D′} (y − Q(s, a; θ))^2
    y = tf.stop_gradient(y)  # θ ← θ − α ∇θL(θ), noting that θ′ is considered a constant with respect to θ
    Z_o = tf.gather(Z_o, A, axis=1, batch_dims=1)

    squared_diff = tf.squared_difference(y, Z_o)
    loss = tf.reduce_sum(squared_diff)

    # Optimiser
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay)
    train = optimizer.minimize(loss)

    return loss, train


def assign_weights(online_scope, target_scope):
    """

    :param online_scope:
    :param target_scope:
    :return:
    """
    online_vars = online_scope.trainable_variables()
    target_vars = target_scope.trainable_variables()

    assigns = [tf.assign(ref, value) for ref, value in zip(target_vars, online_vars)]
    assign = tf.group(assigns)

    return assign


def epsilon_greedy_policy(epsilon, observation, env):
    """

    :param epsilon:
    :param observation:
    :return action:
    """
    if np.random.uniform(0, 1) < (1 - epsilon):
        # at ← arg max_a Q(st, a; θ)
        action = session.run(argmax_Z_o, feed_dict={X_o: observation[np.newaxis, ...]})
    else:
        # at ← random action
        action = env.action_space.sample()

    return action


def moving_average(values, window=30):
    """

    :param values:
    :param window:
    :return moving_average:
    """
    cumsum = np.cumsum(values, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]

    moving_average = cumsum[window - 1:] / values

    return moving_average


def evaluate_model(t, episode, eval_env, f_scrore):
    """

    :param t:
    :param episode:
    :param eval_env:
    :return:
    """
    # You should sum the return obtained across 5 different episodes so that you can compare your results to those
    # listed by Mnih et al. (2015).
    n_play = 30
    n_episode = 5
    score = 0  # sum of returns

    for _ in range(n_play):  # a sequence of 5 episodes is a play
        for _ in range(n_episode):
            eval_observation = eval_env.reset()
            eval_done = False

            while not eval_done:
                # every 100,000 steps (20 times in total), evaluate an ε-greedy policy based on your learned
                # Q-function with ε = 0.001
                eval_action = epsilon_greedy_policy(0.001, eval_observation, eval_env)

                eval_observation, eval_reward, eval_done, eval_info = eval_env.step(eval_action)
                score += eval_reward

    # average the scores across 30 independent plays
    score /= n_play

    # plot the value of the score
    print('Score: ', score)
    f_scrore.write(str(t) + ', ' + str(episode) + ',' + str(score) + '\n')


def train_model(model, session, saver, env, eval_env, replay_buffer, f_train, f_scrore, loss, train, assign, X_o, A,
                R, X_t, Omega, B, done=True, n_steps=2_000_000 + 1, episode=0, total_steps_time=0,
                exploration_steps=1_000_000, s_epsilon=1, f_epsilon=0.1, evaluation=100_000, C=10_000, n=4, ret=0,
                returns=None):

    if returns is None:
        returns = []

    for t in range(n_steps):
        step_start = time.time()
        if t % 1000 == 0:
            print('Step: {}.'.format(t))

        if done:
            # print("Episode finished after {} timesteps".format(t + 1))
            episode += 1
            observation = env.reset()

            # compute the return for the current episode
            returns.append(ret)
            ret = 0

        # env.render()  # render a frame for a certain number of steps

        actual_epsilon = np.interp(t, [0, exploration_steps], [s_epsilon, f_epsilon])

        action = epsilon_greedy_policy(actual_epsilon, observation, env)

        old_observation = observation

        # Obtain the next state and reward by taking action
        observation, reward, done, info = env.step(action)
        ret += reward

        # Store the tuple (st, at, rt+1, st+1, Ωt+1) in the replay buffer D
        replay_buffer.append([old_observation, action, reward, observation, done])

        # monitor the number of steps elapsed, the number of episodes elapsed, and the reward obtained at each step
        f_train.write(str(t) + ', ' + str(episode) + ',' + str(reward) + '\n')

        # The networks are not updated until the replay buffer is populated with M = 10000 transitions.
        if t >= C:
            # Every n = 4 steps, sample a subset/batch D′ ⊂ D composed of B = 32 tuples (transitions) from the replay buffer
            if t % n == 0:
                batch = replay_buffer.sample(B)
                s = np.array(batch[:, 0].tolist())
                a = np.array(batch[:, 1], dtype=np.int)[..., np.newaxis]
                r = np.array(batch[:, 2], dtype=np.float)[..., np.newaxis]
                s1 = np.array(batch[:, 3].tolist())
                omega1 = np.array(batch[:, 4], dtype=np.bool)[..., np.newaxis]

                # Using this batch, update the parameters of the online Q-network to minimize the loss L(θ).
                batch_loss, _ = session.run([loss, train], feed_dict={X_o: s, A: a, R: r, X_t: s1, Omega: omega1})
                if t % 1000 == 0:
                    print('loss ', batch_loss)
                f_train.write(str(t) + ', ' + str(episode) + ',' + str(batch_loss) + '\n')

            # Every C = 10000 steps, copy the parameters of the online network to the target network.
            if t % C == 0:
                session.run(assign)

            # Save a checkpoint
            saver.save(session, 'train/')

            # Evaluation
            if t % evaluation == 0:
                evaluate_model(t, evaluation, eval_env, f_scrore)

        # Estimate the remaining training time based on the average time that each step requires
        step_end = time.time()
        total_steps_time += (step_end - step_start)
        avg_step_time = total_steps_time / (t + 1)

        remaining_steps = n_steps - (t + 1)
        remaining_training_time = avg_step_time * remaining_steps

        if t % 1000 == 0:
            print("Remaining_training_time: {} sec.".format(remaining_training_time))

    # Return per episode, averaged over the last 30 episodes to reduce noise (moving average).
    f_moving_average = open('out/' + model + '/moving_average.txt', "w")
    f_moving_average.write('moving average\n')
    moving_avg = moving_average(returns)

    for el in moving_avg():
        f_moving_average.write(el + '\n')


def main(model, env_name):
    env = wrap_atari_deepmind(env_name, True)
    eval_env = wrap_atari_deepmind(env_name, False)  # the rewards of the evaluation environment should not be clipped

    learning_rate = 0.000_1
    decay = 0.99
    k = env.action_space.n
    gamma = 0.99

    B = 32


    # Initialize replay buffer D, which stores at most M tuples
    replay_buffer = ReplayBuffer()

    # Initialize network parameters θ randomly
    X_o, Z_o, argmax_Z_o, online_scope = net_param('online', k)
    X_t, Z_t, max_Z_t, target_scope = net_param('target', k)

    A = tf.placeholder(tf.int32, [B, 1], name='a')
    R = tf.placeholder(tf.float32, [B, 1], name='r')
    Omega = tf.placeholder(tf.bool, [B, 1], name='omega1')

    loss, train = create_loss(decay, learning_rate, gamma, Z_o, A, R, max_Z_t, Omega)

    assign = assign_weights(online_scope, target_scope)

    session = tf.Session()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("var/tensorboard", session.graph)

    # Training
    check_dir('out/' + model)

    f_train = open('out/' + model + '/train.txt', "w")
    f_train.write('step,episode,reward\n')

    f_loss = open('out/' + model + '/loss.txt', "w")
    f_loss.write('step,episode,loss\n')

    f_scrore = open('out/' + model + '/score.txt', "w")
    f_scrore.write('step,episode,score\n')

    train_model(model, session, saver, env, eval_env, replay_buffer, f_train, f_scrore, loss, train, assign, X_o, A,
                R, X_t, Omega, B)

    # TODO: After training, render one episode of interaction between your agent and the environment.
    #  For this purpose, you may wrap your environment using a gym.wrappers.Monitor.
    #  If your environment object is called env, interacting with the wrapped environment
    #  gym.wrappers.Monitor(env, path, force=True) will cause the corresponding video to be saved to path

    # TODO: Instead of updating the target network every C = 10, 000 steps, experiment with C = 50, 000. Compare in a
    #  single plot the average score across evaluations obtained by these two alternatives.
    #  How do you explain the differences?

    env.close()
    writer.close()
    session.close()


    # TODO: Repeat Steps 4-5 for a different Atari game.

    # TODO: Write your own wrapper for an Atari game. This wrapper should transform observations or rewards in order to
    #  make it much easier for Algorithm 1 to find a high-scoring policy.

    # TODO: Using record agent.py as a guide (see iCorsi3), record your personal gameplay for 10,000 steps.
    #  Use this data to populate the replay buffer, and train your agent for 300, 000 steps.
    #  Compute the average score obtained by the resulting agent.


if __name__ == '__main__':

    main(model='m1', env_name='BreakoutNoFrameskip-v4')
