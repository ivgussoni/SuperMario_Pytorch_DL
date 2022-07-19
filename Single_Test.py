from os.path import join
import time
import numpy as np
import torch

from core.constants import *
from core.model import CNNDQN
from core.wrappers import wrap_environment

MOVS = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]


def test(environment, action_space):
    env = wrap_environment(environment, action_space, monitor=False)
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    model_dir = join(NEW_MODELS, '%s.dat' % environment)
    net.load_state_dict(torch.load(model_dir))

    total_reward = 0.0
    state = env.reset()

    while True:
        time.sleep(0.05)
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        state, reward, done, info = env.step(action)
        env.render()
        # print('Action: {} - Reward: {}'.format(MOVS[action], reward))
        # total_reward += reward
        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            #print(total_reward)
        if done:
            break


if __name__ == '__main__':
    test(ENVIRONMENT, ACTION_SPACE)
