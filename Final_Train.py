from os.path import join

import math
import torch
from torch import save
from torch.optim import Adam

from core.argparser import parse_args
from core.constants import PRETRAINED_MODELS, SAVE_EACH, MEMORY_CAPACITY, NEW_MODELS, LEARNING_RATE
from core.helpers import (compute_td_loss, initialize_models)
from core.replay_buffer import PrioritizedBuffer
from core.test import test
from core.train_information import TrainInformation
from core.wrappers import wrap_environment

args = parse_args()

eps_final = args.epsilon_final
eps_start = args.epsilon_start
decay = args.epsilon_decay

start = args.beta_start
frames = args.beta_frames

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

env = wrap_environment(args.environment, args.action_space)
model, target_model = initialize_models(args.environment, env, device, args.transfer)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
replay_buffer = PrioritizedBuffer(args.buffer_capacity)

complete_times = 0
T_S = 0


def main():
    global T_S, complete_times

    info = TrainInformation()

    for episode in range(args.num_episodes):

        episode_reward = 0.0
        state = env.reset()

        while True:

            epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((episode + 1) / decay))

            if len(replay_buffer) > args.batch_size:
                beta = min(1.0, start + episode * (1.0 - start) / frames)
            else:
                beta = args.beta_start

            action = model.act(state, epsilon, device)

            T_S += 1

            if args.render:
                env.render()

            next_state, reward, done, stats = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            info.update_index()

            if len(replay_buffer) > args.initial_learning:
                if not info.index % args.target_update_frequency:
                    target_model.load_state_dict(model.state_dict())
                optimizer.zero_grad()
                compute_td_loss(model, target_model, replay_buffer, args.gamma, device, args.batch_size, beta)
                optimizer.step()

            if done:

                info.update_rewards(episode_reward)

                if stats['flag_get'] or (episode + 1) % SAVE_EACH == 0:

                    save(model.state_dict(), join(PRETRAINED_MODELS, '%s.dat' % args.environment))

                    record_patch = 'models/model_test/{}/run{}_{}'.format(args.environment, episode + 1,
                                                                          int(round(episode_reward, 0)))

                    flag = test(args.environment, record_patch, args.action_space)

                    if flag:
                        save(model.state_dict(), join(NEW_MODELS, '%s.dat' % args.environment))
                        print('FLAG - TRAINING END')
                        break

                    info.update_best_counter()
                    complete_times += 1

                print(
                    'Epoch: {:d}/{:d} | '
                    'Steps: {:d} | '
                    'Reward: {:.1f} | '
                    'Best Reward: {:.1f} | '
                    'Average: {:.3f} | '
                    'Epsilon: {} | '
                    'LR: {} | '
                    'Flags: {:d} | '
                    'X: {:d} | '
                    'Time: {:d} | '
                    'Memory: {:d}/{:d} | '
                    'Device: {}'.format(
                        episode + 1,
                        args.num_episodes,
                        T_S,
                        round(episode_reward, 3),
                        round(info.best_reward, 3),
                        round(info.average, 3),
                        epsilon,
                        LEARNING_RATE,
                        complete_times,
                        stats['x_pos'],
                        stats['time'],
                        replay_buffer.__len__(),
                        MEMORY_CAPACITY,
                        device))

                break

    env.close()


if __name__ == '__main__':
    main()
