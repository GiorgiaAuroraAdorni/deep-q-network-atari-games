from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

from main import check_dir, moving_average


def plot_step_per_episode(model):
    """

    :param model:
    :return:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/step-per-episode.txt'

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    episodes = lines[:, 1].astype(np.int)

    episode_counter = Counter(episodes)  # number of steps for episode

    mavg_episode_counter = moving_average(np.array(list(episode_counter.values())), window=100)

    plt.figure()

    plt.plot(np.array(list(episode_counter.keys()))[99:], mavg_episode_counter)

    plt.xlabel('episode', fontsize=11)
    plt.ylabel('step', fontsize=11)

    plt.title('Steps per Episode: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'step-per-episode.pdf')
    plt.close()


def plot_return(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/return.txt'

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    steps = lines[:, 0].astype(np.int)
    scores = lines[:, 2].astype(np.float)

    steps = steps.astype(np.float)
    scores = scores.astype(np.float)

    plt.figure()

    plt.plot(steps, scores)

    plt.xlabel('step', fontsize=11)
    plt.ylabel('average score per play', fontsize=11)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(EngFormatter())

    plt.title('Evaluation Return: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'evaluation-return.pdf')
    plt.close()


def plot_loss_moving_average(model):
    """

    :param model:
    :return:
    """

    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/loss.txt'

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    steps = lines[:, 0].astype(np.int)
    losses = lines[:, 2].astype(np.float)

    losses = moving_average(losses, window=50)

    plt.figure()
    plt.plot(steps[50::1000], losses[::1000])

    plt.xlabel('step', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(EngFormatter())

    plt.title('Temporal difference error: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'temporal-difference-error.pdf')
    plt.close()


def plot_return_moving_average(model):
    """

    :param model:
    """
    out_dir = 'out/' + model + '/img/'
    check_dir(out_dir)

    input_dir = 'out/' + model + '/return_moving_average.txt'

    with open(input_dir, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    return_moving_averages = lines[:, 0].astype(np.float)

    episodes = np.arange(30, len(return_moving_averages) + 30)

    plt.figure()

    plt.plot(episodes, return_moving_averages)

    plt.xlabel('episodes', fontsize=11)
    plt.ylabel('return per episode', fontsize=11)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(EngFormatter())

    plt.title('Training Return: "' + model + '"', weight='bold', fontsize=12)
    plt.savefig(out_dir + 'training-return.pdf')
    plt.close()


def plot_return_comparison(model1, model2, label1, label2):
    """

    :param model1:
    :param model2:
    :param label1:
    :parameter label2:
    """
    out_dir = 'out/' + model2 + '/img/comparison/'
    check_dir(out_dir)

    input_dir1 = 'out/' + model1 + '/return.txt'
    input_dir2 = 'out/' + model2 + '/return.txt'

    with open(input_dir1, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    steps1 = lines[:, 0]
    steps1 = steps1.astype(np.float)

    score1 = lines[:, 2]
    score1 = score1.astype(np.float)

    with open(input_dir2, 'r') as f:
        lines = f.readlines()[1:]
        lines = np.array(lines)
        lines = np.char.strip(lines, '\n')
        lines = np.array([x.split(',') for x in lines])

    steps2 = lines[:, 0].astype(np.float)
    score2 = lines[:, 2].astype(np.float)

    plt.figure()

    plt.plot(steps1[:len(steps2)-1], score1[:len(score2)-1], label=label1)
    # plt.plot(steps1, score1, label=label1)
    plt.plot(steps2, score2, label=label2)

    plt.xlabel('episodes', fontsize=11)
    plt.ylabel('score', fontsize=11)
    plt.xlim([-10000, 310000])
    plt.ylim([-1, 30])

    ax = plt.gca()
    ax.xaxis.set_major_formatter(EngFormatter())

    plt.legend()
    plt.title('Evaluation returns of models "' + model1 + '" and "' + model2 + '"', weight='bold',
              fontsize=12)
    plt.savefig(out_dir + 'scores-comparison-' + model1 + '-' + model2 + '.pdf')
    plt.close()


def plot_loss_comparison(models):
    """

    :param models:
    """

    out_dirs = []
    steps = []
    losses = []
    mavg_losses = []

    for i, e in enumerate(models):
        out_dir = 'out/' + e + '/img/comparison/'
        out_dirs.append(out_dir)
        check_dir(out_dir)

        input_dir = 'out/' + e + '/loss.txt'
        with open(input_dir, 'r') as f:
            lines = f.readlines()[1:]
            lines = np.array(lines)
            lines = np.char.strip(lines, '\n')
            lines = np.array([x.split(',') for x in lines])

            steps.append(lines[:, 0].astype(np.int))
            losses.append(lines[:, 2].astype(np.float))
            mavg_losses.append(moving_average(losses[i], window=50))

    plt.figure()
    for i, e in enumerate(models):
        plt.plot(steps[i][50::1000], mavg_losses[i][::1000], label='Loss ' + e)

    plt.xlabel('episodes', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.yaxis.set_major_formatter(EngFormatter())

    plt.legend()
    plt.title('Temporal-difference errors', weight='bold', fontsize=12)
    for i in range(len(models)):
        m = str(models).replace('[', '').replace(']', '').replace(', ', '-').replace("'", "")
        plt.savefig(out_dirs[i] + m + '-losses-comparison.pdf')
    plt.close()


def save_plots():

    plot_step_per_episode(model='m1')
    plot_return(model='m1')
    plot_loss_moving_average(model='m1')
    plot_return_moving_average(model='m1')

    plot_step_per_episode(model='m2')
    plot_return(model='m2')
    plot_loss_moving_average(model='m2')
    plot_return_moving_average(model='m2')

    plot_return_comparison(model1='m1', model2='m2', label1='C = 10 k', label2='C = 50 k')

    plot_step_per_episode(model='m3')
    plot_return(model='m3')
    plot_loss_moving_average(model='m3')
    plot_return_moving_average(model='m3')

    plot_return_comparison(model1='m1', model2='m3', label1='Breakout, C = 10 k', label2='StarGunner')
    plot_return_comparison(model1='m2', model2='m3', label1='Breakout, C = 50 k', label2='StarGunner')

    plot_step_per_episode(model='m4')
    plot_return(model='m4')
    plot_loss_moving_average(model='m4')
    plot_return_moving_average(model='m4')

    plot_return_comparison(model1='m1', model2='m4', label1='C = 10 k, empty buffer', label2='C = 10 k, pre-filled buffer')
    plot_return_comparison(model1='m2', model2='m4', label1='C = 50 k, empty buffer', label2='C = 50 k, pre-filled buffer')

    plot_loss_comparison(['m1', 'm2', 'm3', 'm4'])
    plot_loss_comparison(['m1', 'm2'])
    plot_loss_comparison(['m1', 'm3'])
    plot_loss_comparison(['m1', 'm4'])
    plot_loss_comparison(['m2', 'm4'])
