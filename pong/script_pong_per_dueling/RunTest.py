import time
from itertools import count
import torch

from GetScreen import get_screen
from GetScreen import update_state
import numpy as np
# env, resize, 1, policy_net, device, actions_offset
def test(env, resize, n_episodes, policy, device, actions_offset, render=True,path = F"/../videos"):
    #env = gym.wrappers.Monitor(env, path, force=True)
    total_reward = []
    actions = []
    for episode in range(n_episodes):
        env.reset()
        state = torch.cat((get_screen(env, resize),
                           get_screen(env, resize),
                           get_screen(env, resize),
                           get_screen(env, resize)), dim=1, out=None)
        total_reward.append(0.0)
        actions.append(0.0)
        for t in count():
            state_cuda = state.to(device)
            action = policy(state_cuda).max(1)[1].view(1,1) + actions_offset
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)
            actions[episode] += 1
            total_reward[episode] += reward
            state = update_state(env, resize, state)
            if done:
                break

    env.close()
    print("  - Rewards: {}\n  - Amount of actions: {}".format(n_episodes, total_reward, actions))
    return np.array(total_reward), np.array(actions)