import time
from itertools import count
import torch
import numpy as np

from GetScreen import get_screen
from GetScreen import update_state
import numpy as np
# env, resize, 1, policy_net, device, actions_offset
def test(env, steps_done, resize, n_episodes, policy, device, actions_offset, Skip, render=True,path = F"/../videos"):
    #env = gym.wrappers.Monitor(env, path, force=True)
    print("start")
    total_reward = []
    actions = []
    for episode in range(n_episodes):
        env.reset()
        for j in range(Skip):
            env.step(env.action_space.sample())
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
            if t > 20000:
                actions[episode] = None
                total_reward[episode] = None
                break

            if done:
                break

    env.close()
    actions = np.array(list(filter(None.__ne__, actions)), dtype=int)
    total_reward = np.array(list(filter(None.__ne__, total_reward)), dtype=int)
    print("  - Rewards: {}\n  - Amount of actions: {}".format(total_reward, actions))
    return total_reward, actions