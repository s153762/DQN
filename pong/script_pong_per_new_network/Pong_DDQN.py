import gym
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.optim as optim
import torchvision.transforms as T

from DuelingDQN import DuelingDQN # Get Network
from GetScreen import get_screen
from GetScreen import update_state
from SelectAction import select_action
from PlotDurations import plot_durations
from OptimizeModel import optimize_model
from Per import PER
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

#env_name = "Pong-v0"
#env_name = "PongNoFrameskip-v4"
env_name = "PongDeterministic-v4"
env = gym.make(env_name).unwrapped #

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start using %s\n' % device)

# Display results using tensorboard
init_time = datetime.now()
writer = SummaryWriter(f'../runs/PongDeterministic-v4-per_new_network-less-random-{init_time}_{device}')
print(f"Writing to '../runs/PongDeterministic-v4-per_new_network-less-random-{init_time}_{device}'")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])


# Trainin parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
MEMORY_SIZE = 10000
EPS_DECAY = 100000
TARGET_UPDATE = 100
START_OPTIMIZER = 1000
OPTIMIZE_FREQUENCE = 4
learning_rate = 0.00025

state_cuda = []
batch_cuda = []

n_actions = 3#env.action_space.n
actions_offset = 1



policy_net = DuelingDQN(4, n_actions).to(device)
target_net = DuelingDQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)

memory = PER(MEMORY_SIZE, GAMMA, Transition, device)

model_save_name = 'Pong_POLICY_11.pt'
path = F"../model/{model_save_name}"
torch.save(policy_net.state_dict(), path)

episodes_done = 0
episode_durations = []
steps_done = 0
num_episodes = 3000

for i_episode in range(num_episodes):
    # Initialize the environment and state
    total_reward = 0
    loss = 0
    actions = np.zeros((n_actions), dtype=np.int)
    env.reset()
    for j in range(15):
        env.step(env.action_space.sample())

    state = torch.cat((get_screen(env, resize),
                       get_screen(env, resize),
                       get_screen(env, resize),
                       get_screen(env, resize)), dim=1, out=None)
    next_state = state.clone()

    for t in count():
        # Select and perform an action
        state_cuda = state.to(device)
        steps_done, action, threshold = select_action(steps_done, episodes_done, n_actions, EPS_END, EPS_START, EPS_DECAY, policy_net, device)
        _, reward, done, _ = env.step(action.item() + actions_offset)
        total_reward += reward
        actions[action.item()] += 1

        reward = torch.tensor([reward], device=device)

        # Observe new state
        next_state = update_state(env, resize, state, done)

        # Store the transition in memory
        memory.push(state, action, reward, next_state, done, policy_net, target_net)

        # Move to the next state
        if next_state is not None:
            state = next_state
        else:
            state = None

        # Perform one step of the optimization (on the target network)
        if (steps_done % OPTIMIZE_FREQUENCE == 0) and steps_done > START_OPTIMIZER:
            temp = optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer)

            if temp is not None:
                loss += temp

        if done:
            episodes_done += 1
            episode_durations.append(t + 1)
            # plot_durations()
            break

        if steps_done % TARGET_UPDATE == 0 and steps_done > (START_OPTIMIZER +TARGET_UPDATE):
            target_net.load_state_dict(policy_net.state_dict())



    # plot data
    writer.add_scalar('training loss', loss, i_episode)
    writer.add_scalar('total reward', total_reward, i_episode)
    for i in range(len(actions)):
        writer.add_scalars('Actions',{str(i):actions[i]}, i_episode)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("Epoch: ", i_episode, " - Total reward: ", total_reward, "Episode duration: ", episode_durations[-1],
              "Actions: ", actions, "Threshold: ", threshold)
        torch.save(policy_net.state_dict(), path)

    if i_episode % 250 == 0:
        print("Model new iteration Saved %d" % (i_episode))
        torch.save(policy_net.state_dict(), path.replace(".pt", F"_{i_episode}.pt"))

    del state_cuda
    torch.cuda.empty_cache()

torch.save(policy_net.state_dict(), path)
print('Complete')
plot_durations(episode_durations)
env.close()

