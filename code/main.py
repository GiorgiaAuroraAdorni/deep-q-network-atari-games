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
with open('/proc/self/oom_score_adj', 'w') as f:
    f.write('1000\n')

### EXAMPLE ###

# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()  # start new episode
#     for t in range(100):
#         print(observation)
#         env.render()  # render a frame for a certain number of steps
#         observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
# env.close()

###############


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
    :return A_conv, W_conv
    """
    # Initialize the weights with tf.variance_scaling_initializer
    W_init = tf.variance_scaling_initializer()
    W_conv = tf.Variable(W_init([filter_size, filter_size, input_size, output_size]))

    # Initialize the biases with tf.zeros initializer
    b_init = tf.zeros_initializer()
    b_conv = tf.Variable(b_init(output_size))

    A_conv = tf.nn.conv2d(input, W_conv, strides=[1, stride, stride, 1], padding='SAME') + b_conv

    # Apply activation function
    A_conv = tf.nn.relu(A_conv)

    return A_conv, W_conv


def create_fc_layer(input_size, output_size, input, activation):
    """

    :param input_size: input size
    :param output_size: number of neurons
    :param input: input of the layer
    :param activation: boolean parameter that if is True the rectified linear activations is applied
    :return A_fc
    """
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
    A_conv1, W_conv1 = create_conv_layer(8, 4, input_shape, 32, X)

    # Convolutional layer 2: 64 filters, 4 × 4.
    A_conv2, W_conv2 = create_conv_layer(4, 2, 32, 64, A_conv1)

    # Convolutional layer 3: 64 filters, 3 × 3.
    A_conv3, W_conv3 = create_conv_layer(3, 1, 64, 64, A_conv2)

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
            out = tf.reduce_max(Z, axis=1)

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
    Z_o = tf.gather(Z_o, A, axis=1)

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


################################################################################


env_name = 'BreakoutNoFrameskip-v4'
clip_rewards = True

env = wrap_atari_deepmind(env_name, clip_rewards)

learning_rate = 0.0001
decay = 0.99
k = env.action_space.n

gamma = 0.99  # discount factor
n_steps = 2000000
s_epsilon = 1  # starting exploration rate
f_epsilon = 0.1  # final exploration rate
exploration_steps = 1000000

C = 10000
n = 4
B = 32

# Initialize replay buffer D, which stores at most M tuples
replay_buffer = ReplayBuffer()

# Initialize network parameters θ randomly
X_o, Z_o, argmax_Z_o, online_scope = net_param('online', k)
X_t, Z_t, max_Z_t, target_scope = net_param('target', k)

A = tf.placeholder(tf.int32, [B], name='a')
R = tf.placeholder(tf.float32, [B], name='r')
Omega = tf.placeholder(tf.bool, [B], name='omega1')

loss, train = create_loss(decay, learning_rate, gamma, Z_o, A, R, max_Z_t, Omega)

# Avoid allocating all GPU memory upfront.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

# Training
done = True

for t in range(n_steps):
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        observation = env.reset()

    # env.render()  # render a frame for a certain number of steps

    actual_epsilon = np.interp(t, [0, exploration_steps], [s_epsilon, f_epsilon])

    if np.random.uniform(0, 1) < (1 - actual_epsilon):
        # at ← arg max_a Q(st, a; θ)
        action = session.run(argmax_Z_o, feed_dict={X_o: observation[np.newaxis, ...]})
    else:
        # at ← random action
        action = env.action_space.sample()

    old_observation = observation

    # Obtain the next state and reward by taking action at
    observation, reward, done, info = env.step(action)

    # Store the tuple (st, at, rt+1, st+1, Ωt+1) in the replay buffer D
    replay_buffer.append([old_observation, action, reward, observation, done])

    # The networks are not updated until the replay buffer is populated with M = 10000 transitions.
    if t >= C:
        # Every n = 4 steps, sample a subset/batch D′ ⊂ D composed of B = 32 tuples (transitions) from the replay buffer
        if t % n == 0:
            batch = replay_buffer.sample(B)
            s = np.array(batch[:, 0].tolist())
            a = np.array(batch[:, 1], dtype=np.int)
            r = np.array(batch[:, 2], dtype=np.float)
            s1 = np.array(batch[:, 3].tolist())
            omega1 = np.array(batch[:, 4], dtype=np.bool)

            # Using this batch, update the parameters of the online Q-network to minimize the loss L(θ).
            batch_loss = session.run(loss, feed_dict={X_o: s, A: a, R: r, X_t: s1, Omega: omega1})
            print(batch_loss)
    # Every C = 10000 steps, copy the parameters of the online network to the target network.
    if t % C == 0:
        assign = assign_weights(online_scope, target_scope)

env.close()




