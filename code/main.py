import os
import time
from collections import Counter

import gym
import matplotlib.pyplot as plt
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
# with open('/proc/self/oom_score_adj', 'w') as f:
#     f.write('1000\n')


class ReplayBuffer(object):
    def __init__(self, counter=0, capacity=10000):
        self.counter = counter
        self.capacity = capacity
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

    def is_full(self):
        """

        :return:
        """
        if self.buffer[-1].all() == np.empty([1, 5], dtype=object).all():
            return False
        else:
            return True


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
    :param k:
    :return X, Z, out, scope
    """
    with tf.variable_scope("model_{}".format(model)) as scope:
        X = tf.placeholder(tf.float32, [None, 84, 84, 4], name='X')

        Z = conv_net(X, k)
        if model == 'online':
            out = tf.argmax(Z, axis=1)
        else:
            out = tf.reduce_max(Z, axis=1, keepdims=True)

    return X, Z, out, scope


def create_loss(Z_o, A, R, max_Z_t, Omega, decay=0.99, learning_rate=0.000_1, gamma=0.99):
    """

    :param Z_o:
    :param A:
    :param R:
    :param max_Z_t:
    :param Omega:
    :param decay:
    :param learning_rate:
    :param gamma:
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


def epsilon_greedy_policy(session, argmax_Z_o, X_o, epsilon, observation, env):
    """

    :param session:
    :param argmax_Z_o:
    :param X_o:
    :param epsilon:
    :param observation:
    :param env:
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

    moving_average = cumsum[window - 1:] / window

    return moving_average


def evaluate_model(session, argmax_Z_o, X_o, t, episode, eval_env, f_score):
    """

    :param session:
    :param argmax_Z_o:
    :param X_o:
    :param t:
    :param episode:
    :param eval_env:
    :param f_scrore:
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
                eval_action = epsilon_greedy_policy(session, argmax_Z_o, X_o, 0.001, eval_observation, eval_env)

                eval_observation, eval_reward, eval_done, eval_info = eval_env.step(eval_action)
                score += eval_reward

    # average the scores across 30 independent plays
    score /= n_play

    # plot the value of the score
    print('Score: ', score)
    f_score.write(str(t) + ', ' + str(episode) + ',' + str(score) + '\n')


def train_model(model, session, saver, env, eval_env, replay_buffer, f_step_episode, f_reward, f_loss, f_score, loss,
                train, assign, X_o, A, R, X_t, Omega, B, argmax_Z_o, C, n_steps, done=True, episode=0,
                total_steps_time=0, exploration_steps=1_000_000, s_epsilon=1, f_epsilon=0.1, evaluation=100_000, n=4,
                ret=0.0, returns=None):
    """

    :param model:
    :param session:
    :param saver:
    :param env:
    :param eval_env:
    :param replay_buffer:
    :param f_reward:
    :param f_loss:
    :param f_score:
    :param loss:
    :param train:
    :param assign:
    :param X_o:
    :param A:
    :param R:
    :param X_t:
    :param Omega:
    :param B:
    :param argmax_Z_o:
    :param C:
    :param done:
    :param n_steps:
    :param episode:
    :param total_steps_time:
    :param exploration_steps:
    :param s_epsilon:
    :param f_epsilon:
    :param evaluation:
    :param n:
    :param ret:
    :param returns:
    :return:
    """

    if returns is None:
        returns = []

    for t in range(n_steps):
        step_start = time.time()
        f_step_episode.write(str(t) + ', ' + str(episode) + '\n')

        if t % 1000 == 0:
            print('Step: {}.'.format(t))

        if done:
            # print("Episode finished after {} timesteps".format(t + 1))
            episode += 1
            observation = env.reset()

            # compute the return for the current episode
            returns.append(ret)
            ret = 0.0

        # env.render()  # render a frame for a certain number of steps

        actual_epsilon = np.interp(t, [0, exploration_steps], [s_epsilon, f_epsilon])

        action = epsilon_greedy_policy(session, argmax_Z_o, X_o, actual_epsilon, observation, env)

        old_observation = observation

        # Obtain the next state and reward by taking action
        observation, reward, done, info = env.step(action)
        ret += reward
        if reward > 0:
            print()

        # Store the tuple (st, at, rt+1, st+1, Ωt+1) in the replay buffer D
        replay_buffer.append([old_observation, action, reward, observation, done])

        # monitor the number of steps elapsed, the number of episodes elapsed, and the reward obtained at each step
        f_reward.write(str(t) + ', ' + str(episode) + ',' + str(reward) + '\n')

        # The networks are not updated until the replay buffer is populated with M = 10000 transitions.
        if replay_buffer.is_full():
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
                f_loss.write(str(t) + ', ' + str(episode) + ',' + str(batch_loss) + '\n')

            # Every C = 10000 steps, copy the parameters of the online network to the target network.
            if t % C == 0:
                session.run(assign)

                # Save a checkpoint
                print('Saving model…')
                checkpoit_dir = 'out/' + model + '/train/train.ckpt'

                saver.save(session, checkpoit_dir)

            # Evaluation
            if t % evaluation == 0:
                print('Evaluation…')
                evaluate_model(session, argmax_Z_o, X_o, t, evaluation, eval_env, f_score)

        # Estimate the remaining training time based on the average time that each step requires
        step_end = time.time()
        total_steps_time += (step_end - step_start)
        avg_step_time = total_steps_time / (t + 1)

        remaining_steps = n_steps - (t + 1)
        remaining_training_time = avg_step_time * remaining_steps

        if t % 1000 == 0:
            print("Remaining_training_time: {} sec.".format(remaining_training_time))

    f_reward.close()
    f_score.close()
    f_loss.close()

    # Return per episode, averaged over the last 30 episodes to reduce noise (moving average).
    f_moving_average = open('out/' + model + '/moving_average.txt', "w")
    f_moving_average.write('moving average\n')
    moving_avg = moving_average(returns)

    for el in moving_avg:
        f_moving_average.write(str(el) + '\n')

    f_moving_average.close()


def test_model(X_o, argmax_Z_o, eval_env, session, video_dir):
    """

    :param X_o:
    :param argmax_Z_o:
    :param eval_env:
    :param session:
    :param video_dir:
    :return test_env:
    """
    # Wrap the environment using a gym.wrappers.Monitor.
    test_env = gym.wrappers.Monitor(eval_env, video_dir, video_callable=lambda _: True, mode="evaluation", force=True)

    for _ in range(10):
        test_observation = test_env.reset()
        test_done = False

        while not test_done:
            test_env.render()
            test_action = epsilon_greedy_policy(session, argmax_Z_o, X_o, 0.001, test_observation, test_env)

            test_observation, test_reward, test_done, test_info = test_env.step(test_action)

    return test_env


def main(model, env_name, do_train, C=10_000, n_steps=2_000_000 + 1):
    """

    :param model:
    :param env_name:
    :param do_train:
    :param C:
    :return:
    """
    env = wrap_atari_deepmind(env_name, True)
    eval_env = wrap_atari_deepmind(env_name, False)  # the rewards of the evaluation environment should not be clipped

    k = env.action_space.n
    B = 32

    # Initialize replay buffer D, which stores at most M tuples
    replay_buffer = ReplayBuffer()
    # TODO: Using record agent.py as a guide, record your personal gameplay for 10,000 steps.
    #  Use this data to populate the replay buffer, and train your agent for 300,000 steps.
    #  Compute the average score obtained by the resulting agent.
    array = np.load('replay_buffer_FOR_STUDENTS.npy')

    # Initialize network parameters θ randomly
    X_o, Z_o, argmax_Z_o, online_scope = net_param('online', k)
    X_t, Z_t, max_Z_t, target_scope = net_param('target', k)

    A = tf.placeholder(tf.int32, [B, 1], name='a')
    R = tf.placeholder(tf.float32, [B, 1], name='r')
    Omega = tf.placeholder(tf.bool, [B, 1], name='omega1')

    loss, train = create_loss(Z_o, A, R, max_Z_t, Omega)

    assign = assign_weights(online_scope, target_scope)

    session = tf.Session()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("var/tensorboard", session.graph)

    # Training
    if do_train:
        check_dir('out/' + model)

        f_step_episode = open('out/' + model + '/step-per-episode.txt', "w")
        f_step_episode.write('step,episode\n')

        f_reward = open('out/' + model + '/reward.txt', "w")
        f_reward.write('step,episode,reward\n')

        f_loss = open('out/' + model + '/loss.txt', "w")
        f_loss.write('step,episode,loss\n')

        f_score = open('out/' + model + '/score.txt', "w")
        f_score.write('step,episode,score\n')

        train_model(model, session, saver, env, eval_env, replay_buffer, f_step_episode, f_reward, f_loss, f_score,
                    loss, train, assign, X_o, A, R, X_t, Omega, B, argmax_Z_o, C, n_steps)

    # After training, render one episode of interaction between your agent and the environment.
    print('\nRendering one episode of interaction between the agent and the environment…')
    video_dir = 'out/' + model + '/video'
    check_dir(video_dir)

    test_env = test_model(X_o, argmax_Z_o, eval_env, session, video_dir)

    env.close()
    eval_env.close()
    test_env.close()
    writer.close()
    session.close()

    # TODO: Write your own wrapper for an Atari game. This wrapper should transform observations or rewards in order to
    #  make it much easier for Algorithm 1 to find a high-scoring policy.


def plot_step_per_episode(model):
    """

    :param model:
    :return:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/step-per-episode.txt'

    episodes = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            # step = np.append(step, float(el[0]))
            episodes = np.append(episodes, float(el[1]))

    episode_counter = Counter(episodes)

    plt.xlabel('step', fontsize=11)
    plt.ylabel('episode', fontsize=11)

    plt.plot(list(episode_counter.keys()), list(episode_counter.values()), label='Steps per Episode')
    plt.title('Steps per Episode: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'step-per-episode.pdf')
    plt.show()
    plt.close()


def plot_score(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/score.txt'

    steps = []
    scores = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            steps = np.append(steps, float(el[0]))
            scores = np.append(scores, float(el[2]))

    plt.xlabel('step', fontsize=11)
    plt.ylabel('score', fontsize=11)

    plt.plot(steps, scores, label='Score')
    plt.title('Score: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'score.pdf')
    plt.show()
    plt.close()


def plot_reward(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/reward.txt'

    steps = []
    rewards = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            steps = np.append(steps, float(el[0]))
            rewards = np.append(rewards, float(el[2]))

    plt.xlabel('step', fontsize=11)
    plt.ylabel('reward', fontsize=11)

    plt.plot(steps, rewards, label='Reward')
    plt.title('Reward: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'reward.pdf')
    plt.show()
    plt.close()


def plot_loss(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/loss.txt'

    steps = []
    losses = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            steps = np.append(steps, float(el[0]))
            losses = np.append(losses, float(el[2]))

    plt.xlabel('step', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.plot(steps, losses, label='Loss')
    plt.title('Loss: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'loss.pdf')
    plt.show()
    plt.close()


def plot_moving_average(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/moving_average.txt'

    moving_averages = []

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            moving_averages = np.append(moving_averages, float(el[0]))

    episodes = np.arange(30, len(moving_averages) + 30)

    plt.xlabel('episodes', fontsize=11)
    plt.ylabel('moving-average', fontsize=11)

    plt.plot(episodes, moving_averages, label='Moving Average')
    plt.title('Moving Average: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'moving-average.pdf')
    plt.show()
    plt.close()


def plot_score_comparison(model1, model2):
    """

    :param model1:
    :param model2:
    """
    out_dir = 'out/' + model2 + '/img/comparison/'
    check_dir(out_dir)

    input_dir1 = 'out/' + model1 + '/score.txt'
    input_dir2 = 'out/' + model2 + '/score.txt'

    score1 = []
    score2 = []
    step = []

    with open(input_dir1, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            step = np.append(step, float(el[0]))
            score1 = np.append(score1, float(el[2]))

    with open(input_dir2, 'r') as f:
        lines = f.readlines()[1:]

        for l in lines:
            el = l.strip('\n').split(',')
            score2 = np.append(score2, float(el[2]))

    plt.xlabel('episodes', fontsize=11)
    plt.ylabel('score', fontsize=11)

    plt.plot(step, score1, label='Score' + model1)
    plt.plot(step, score2, label='Score' + model2)

    plt.title('Score comparison among model "' + model1 + '" and "' + model2 + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'scores-comparison-' + model1 + '-' + model2 + '.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':

    main(model='m1', env_name='BreakoutNoFrameskip-v4', do_train=True, n_steps=101_000 + 1)

    # plot_step_per_episode(model='m1')
    # # plot_score(model='m1')
    # plot_reward(model='m1')
    # plot_loss(model='m1')
    # plot_moving_average(model='m1')

    # Instead of updating the target network every C = 10,000 steps, experiment with C = 50,000.
    # main(model='m2', env_name='BreakoutNoFrameskip-v4', do_train=False, C=50_000)

    # plot_step_per_episode(model='m2')
    # # plot_score(model='m2')
    # plot_reward(model='m2')
    # plot_loss(model='m2')
    # plot_moving_average(model='m2')

    # Compare in a single plot the average score across evaluations obtained by these two alternatives.
    # How do you explain the differences?
    # plot_score_comparison(model1='m1', model2='m2')

    # Experiment with a different Atari game.
    # main(model='m3', env_name='StarGunnerNoFrameskip-v4', do_train=False)

    # plot_step_per_episode(model='m3')
    # # plot_score(model='m3')
    # plot_reward(model='m3')
    # plot_loss(model='m3')
    # plot_moving_average(model='m3')
    # plot_score_comparison(model1='m1', model2='m3')
    # plot_score_comparison(model1='m2', model2='m3')

    main(model='m4', env_name='BreakoutNoFrameskip-v4', do_train=False, n_steps=300_000)

    # plot_step_per_episode(model='m4')
    # # plot_score(model='m4')
    # plot_reward(model='m4')
    # plot_loss(model='m4')
    # plot_moving_average(model='m4')
    # plot_score_comparison(model1='m1', model2='m4')
    # plot_score_comparison(model1='m2', model2='m4')
