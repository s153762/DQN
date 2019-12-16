import time
from itertools import count

import gym
import torch

from GetScreen import get_screen
from GetScreen import update_state
# env, resize, 1, policy_net, device, actions_offset
def test(env, resize, n_episodes, policy, device, actions_offset, render=True,path = F"/../videos"):
    #env = gym.wrappers.Monitor(env, path, force=True)
    state = torch.cat((get_screen(env, resize),
                       get_screen(env, resize),
                       get_screen(env, resize),
                       get_screen(env, resize)), dim=1, out=None)
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        for t in count():
            state_cuda = state.to(device)
            action = policy(state_cuda).max(1)[1].view(1,1) + actions_offset
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward
            state = update_state(env, resize, state)
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return total_reward