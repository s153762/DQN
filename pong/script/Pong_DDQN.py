import gym
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import numpy as np

import torch
import torch.optim as optim
import torchvision.transforms as T

from ReplayMemory import ReplayMemory # Get ReplayMemory
from DQN import DQN # Get Network
from GetScreen import get_screen
from SelectAction import select_action
from PlotDurations import plot_durations
from OptimizeModel import optimize_model

env = gym.make('Pong-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start using %s\n' % device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(80, interpolation=Image.CUBIC),
                    T.ToTensor()])


env.reset()
plt.figure()
plt.imshow(get_screen(env, resize, device).cpu().squeeze(0).squeeze(0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.savefig("../plt/ExampleExtractedScreen")

BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
learning_rate = 3e-4
decay_rate = 0.99

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, weight_decay=decay_rate)
memory = ReplayMemory(10000, Transition)

model_save_name = 'Pong_POLICY_3_.pt'
path = F"../model/{model_save_name}"
torch.save(policy_net.state_dict(), path)

episode_durations = []

num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    # last_screen = get_screen(env, resize, device)
    # current_screen = get_screen(env, resize, device)
    # state = current_screen - last_screen
    state = torch.cat((get_screen(env, resize, device),
                       get_screen(env, resize, device),
                       get_screen(env, resize, device),
                       get_screen(env, resize, device)), dim=1, out=None)
    next_state = state.clone()

    total_reward = 0
    counter = 0
    actions = np.zeros((3))
    for t in count():
        # Select and perform an action
        action, threshold = select_action(state, EPS_END, EPS_START, EPS_DECAY, policy_net, device)
        actions[action.item() - 1] += 1
        _, reward, done, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        current_screen = get_screen(env, resize, device)
        if not done:
            next_state[:, 1:4, :, :] = next_state[:, 0:3, :, :].clone()
            next_state[:, 0, :, :] = current_screen
            # next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if counter == 4:
            counter = 0
            optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer)
        else:
            counter += 1
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

    # Update the target network
    print("Epoch: ", i_episode, " - Total reward: ", total_reward, "Episode duration: ", episode_durations[-1],
          "Actions: ", actions, "Threshold: ", threshold)
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 2 == 0:
        torch.save(policy_net.state_dict(), path)
        print("Model Saved %d" % (i_episode))
        if i_episode % 5000 == 0:
            torch.save(policy_net.state_dict(), path.replace(".pt", F"{i_episode}.pt"))

torch.save(policy_net.state_dict(), path)
print('Complete')
plot_durations(episode_durations)
env.render()
env.close()
plt.ioff()
