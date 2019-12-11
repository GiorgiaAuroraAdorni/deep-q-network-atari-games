import gym

### EXAMPLE ###
from atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()  # start new episode
    for t in range(100):
        print(observation)
        env.render()  # render a frame for a certain number of steps
        observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()

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

    A_conv3_shapes = X.shape.as_list()
    A_conv3_units = np.product(A_conv3_shapes[1:])
    A_conv3_flat = tf.reshape(A_conv3, [A_conv3_shapes[0], A_conv3_units])

    # Fully connected layer 1: 512 units.
    A_fc1 = create_fc_layer(A_conv3_units, 512, A_conv3_flat, True)

    # Fully connected layer 2: k units.
    A_fc2 = create_fc_layer(512, k, A_fc1, False)

    return A_fc2


def net_param(model, learning_rate, decay, neurons_fc):
    """

    :param model: current model
    :param learning_rate: learning rate of the model
    :param neurons_fc: number of units for the fully connected layer
    :return X, Y, Z, loss, accuracy, train
    """
    with tf.variable_scope("model_{}".format(model)):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')

        Z = conv_net(X, neurons_fc)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
        loss = tf.reduce_mean(loss)

        hits = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

        # Optimiser
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay)
        train = optimizer.minimize(loss)

    return X, Y, Z, loss, accuracy, train


class ReplayBuffer(object):
    def __init__(self):
        self.counter = 0
        self.capacity = 10000
        self.buffer = np.zeros([self.capacity, 5])

    def append(self, data):
        """

        :param data:
        :return:
        """
        self.buffer[self.counter] = data
        self.counter = np.mod([self.counter + 1, self.capacity])

    def sample(self, samples, datas):
        """
        :return:
        """
        idx = np.random.choice(self.capacity, samples)
        return datas[idx]


env_name = 'BreakoutNoFrameskip-v4'
clip_rewards = True

env = wrap_atari_deepmind(env_name, clip_rewards)

