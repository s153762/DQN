import gym
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import numpy as np
import torch.nn as nn

import torch
import torch.optim as optim
import torchvision.transforms as T

from ReplayMemory import ReplayMemory # Get ReplayMemory
from DQN import DQN # Get Network
from GetScreen import get_screen
from SelectAction import select_action
from PlotDurations import plot_durations
from OptimizeModel import optimize_model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

env = gym.make('Pong-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start using %s\n' % device)

# Display results using tensorboard
init_time = datetime.now()
writer = SummaryWriter(f'runs/Pong-v0_{init_time}_{device}')
print(f"Writing to 'runs/Pong-v0_{init_time}_{device}'")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])

env.reset()
plt.figure()
plt.imshow(get_screen(env, resize, device).cpu().squeeze(0).squeeze(0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.savefig("../plt/ExampleExtractedScreen")

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
MEMORY_SIZE = 50000
EPS_DECAY = 1000000 + MEMORY_SIZE
TARGET_UPDATE = 10000
OPTIMIZE_FREQUENCE = 4
learning_rate = 3e-4
decay_rate = 0.01

n_actions = env.action_space.n
actions_offset = 1

criterion = nn.SmoothL1Loss().cuda() # Not used right now

policy_net = DQN(4, n_actions).to(device)
target_net = DQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, alpha=0.9, eps=1e-02, momentum=0.0)
memory = ReplayMemory(MEMORY_SIZE, Transition)

model_save_name = 'Pong_POLICY_5_.pt'
path = F"../model/{model_save_name}"
torch.save(policy_net.state_dict(), path)

episode_durations = []

num_episodes = 1000000
counter = 1
loss = list()

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    for j in range(22):
        env.step(env.action_space.sample())
    # last_screen = get_screen(env, resize, device)
    # current_screen = get_screen(env, resize, device)
    # state = current_screen - last_screen
    state = torch.cat((get_screen(env, resize, device),
                       get_screen(env, resize, device),
                       get_screen(env, resize, device),
                       get_screen(env, resize, device)), dim=1, out=None)
    next_state = state.clone()

    total_reward = 0
    actions = np.zeros((n_actions), dtype=np.int)
    temp = None
    for t in count():
        # Select and perform an action
        action, threshold = select_action(state, n_actions, EPS_END, EPS_START, EPS_DECAY, policy_net, device)
        Q = action.item()
        actions[action.item()] += 1
        _, reward, done, _ = env.step(action.item() + actions_offset)
        total_reward += reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        current_screen = get_screen(env, resize, device)
        if not done:
            next_state[:, 1:4, :, :] = next_state[:, 0:3, :, :].clone()
            next_state[:, 0, :, :] = current_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        if next_state is not None:
            state = next_state.clone()
        else:
            state = None

        # Perform one step of the optimization (on the target network)
        if (counter % OPTIMIZE_FREQUENCE == 0) and counter > 10000:
            temp = optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer)

        if temp is not None:
            loss.append(temp.item())

        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

        if counter % TARGET_UPDATE == 0 and counter > 10000:
            target_net.load_state_dict(policy_net.state_dict())

        counter += 1
    # Update the target network
    print("Epoch: ", i_episode, " - Total reward: ", total_reward, "Episode duration: ", episode_durations[-1], "Actions: ", actions, "Threshold: ", threshold)
    writer.add_scalar('training loss', np.sum(loss), i_episode)
    loss = list()
    writer.add_scalar('total reward', total_reward, i_episode)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 100 == 0:
        torch.save(policy_net.state_dict(), path)
        print("Model Saved %d" % (i_episode))
        if i_episode % 10000 == 0:
            torch.save(policy_net.state_dict(), path.replace(".pt", F"{i_episode}.pt"))

torch.save(policy_net.state_dict(), path)
print('Complete')
plot_durations(episode_durations)
env.close()

