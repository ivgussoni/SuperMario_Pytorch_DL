from os.path import join

import numpy as np
import torch

from core.constants import *
from core.model import CNNDQN
from core.wrappers import wrap_environment


def test(environment, record_patch, action_space):
    flag = False
    env = wrap_environment(environment, action_space, record_patch, monitor=True)
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    model_dir = join(PRETRAINED_MODELS, '%s.dat' % environment)
    net.load_state_dict(torch.load(model_dir))

    total_reward = 0.0
    state = env.reset()
    while True:
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if info['flag_get']:
            #print('WE GOT THE FLAG!!!!!!!')
            flag = True
        if done:
            break

    return flag
