import time
import os
import numpy as np
import torch

from torch import save
from core.constants import *
from core.model import CNNDQN
from core.wrappers import wrap_environment
from os.path import join


def test(environment, action_space):
    flag = False
    env = wrap_environment(environment, action_space, monitor=False)
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(join(PRETRAINED_MODELS, '%s.dat' % environment)))

    total_reward = 0.0
    state = env.reset()

    while True:
        #time.sleep(0.025)
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:

            if info['flag_get']:
                save(net.state_dict(), join(NEW_MODELS, '%s.dat' % environment))
                flag = True
            else:
                save(net.state_dict(), join(UNCOMPLETED_MODELS, '%s.dat' % environment))

            break

    env.close()
    return flag


if __name__ == '__main__':
    for filename in os.listdir(PRETRAINED_MODELS):
        test(str(os.path.splitext(filename)[0]), ACTION_SPACE)
