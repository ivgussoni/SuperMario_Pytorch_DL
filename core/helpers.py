import math
import numpy as np
import torch
from .constants import PRETRAINED_MODELS
from .model import CNNDQN
from os.path import join
from torch import FloatTensor, LongTensor
from torch.autograd import Variable


class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end


def compute_td_loss(model, target_net, replay_buffer, gamma, device, batch_size, beta):
    batch = replay_buffer.sample(batch_size, beta)
    state, action, reward, next_state, done, indices, weights = batch

    state = Variable(FloatTensor(np.float32(state))).to(device)
    next_state = Variable(FloatTensor(np.float32(next_state))).to(device)
    action = Variable(LongTensor(action)).to(device)
    reward = Variable(FloatTensor(reward)).to(device)
    done = Variable(FloatTensor(done)).to(device)
    weights = Variable(FloatTensor(weights)).to(device)

    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())


def load_model(environment, model, target_model):
    model_name = join(PRETRAINED_MODELS, '%s.dat' % environment)
    model.load_state_dict(torch.load(model_name))
    target_model.load_state_dict(model.state_dict())
    return model, target_model


def initialize_models(environment, env, device, transfer):
    model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)
    if transfer:
        model, target_model = load_model(environment, model, target_model)
    return model, target_model
