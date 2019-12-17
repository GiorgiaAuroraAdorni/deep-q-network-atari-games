import signal
import time
import sys
import tty
import termios
import numpy as np
from main import wrap_atari_deepmind


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def raise_timeout(signum, frame):
    raise TimeoutError()

timeout = 0.15

signal.signal(signal.SIGALRM, raise_timeout)

env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', clip_rewards=True)

REPLAY_BUFFER_PATH = 'replay_buffer_FOR_STUDENTS'
mem_size = 10000
n_feat = np.prod(np.array(env.observation_space.shape))
replay_buffer = np.zeros((mem_size, 3 + n_feat * 2), dtype=np.uint8)

step = 0
print("4-LEFT, 6-RIGHT, 8-FIRE, q-EXIT.")
print("Note the inputs should be in the CONSOLE, NOT THE RENDERED ENV.")
while step < mem_size:

    observation = env.reset()
    done = False

    # Run one episode
    while not done:
        print('Step [%d/%d]' % (step, mem_size))
        
        signal.setitimer(signal.ITIMER_REAL, timeout, 0)
        action = 0
        try:
            key = getch()
            if key == '4':
                action = 3  # left
            elif key == '6':
                action = 2  # right
            elif key == '8':
                action = 1  # fire
            elif key == 'q':
                sys.exit()
            
            time.sleep(timeout)
        except TimeoutError:
            pass

        next_observation, reward, done, info = env.step(action)
        env.render()

        # store transition
        s = np.uint8(np.asarray(observation)*255)
        a = np.uint8(action)
        r = np.uint8(reward)
        d = np.uint8(done)
        next_s = np.uint8(np.asarray(next_observation)*255)

        transition = np.hstack(
            (np.reshape(s, [-1]), [a, r, int(d)], np.reshape(next_s, [-1])))
        replay_buffer[step, :] = transition

        observation = next_observation

        step+=1
        if step == mem_size:
            done = True

np.save(REPLAY_BUFFER_PATH, replay_buffer)
print('Saved the replay buffer to', REPLAY_BUFFER_PATH)