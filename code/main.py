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

