import sys
import os
import gym
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T
from datetime import datetime, time
from torch.utils.tensorboard import SummaryWriter

from ReplayMemory import ReplayMemory # Get ReplayMemory
from DQN import DQN # Get Network

sys.path.append(os.path.abspath('../script_common'))
from OptimizeModel import optimize_model
from GetScreen import get_screen
from GetScreen import update_state
from SelectAction import select_action
from PlotDurations import plot_durations
from RunTest import test
sys.path.append(os.path.abspath('../script_pong'))


env_name = "PongDeterministic-v4"
env = gym.make(env_name).unwrapped
envTest = gym.make(env_name).unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start using %s\n' % device)

# Display results using tensorboard
init_time = datetime.now()
name = f'Baseline_{init_time}'
path = f'../runs/report_runs/{name}'
writer = SummaryWriter(path)
print(f"Writing to '{path}'")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Trainin parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
MEMORY_SIZE = 10000
EPS_DECAY = 100000
TARGET_UPDATE = 10000
START_OPTIMIZER = 1000
OPTIMIZE_FREQUENCE = 4
RUN_TEST = 2500
learning_rate = 0.00025

state_cuda = []
batch_cuda = []

n_actions = 3 #env.action_space.n
actions_offset = 1

policy_net = DQN(4, n_actions).to(device)
target_net = DQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)

memory = ReplayMemory(MEMORY_SIZE, Transition)

model_save_name = f'{name}.pt'
path = F"../models/report_models/{model_save_name}"
torch.save(policy_net.state_dict(), path)

episodes_done = 0
steps_done = 0
episode_durations = []
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
        steps_done, action, threshold = select_action(steps_done, state, n_actions, EPS_END, EPS_START, EPS_DECAY, policy_net, device)
        _, reward, done, _ = env.step(action.item() + actions_offset)
        total_reward += reward
        actions[action.item()] += 1

        reward = torch.tensor([reward], device=device)

        # Observe new state
        next_state = update_state(env, resize, state, done)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

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
            break

        if steps_done % TARGET_UPDATE == 0 and steps_done > (START_OPTIMIZER + TARGET_UPDATE):
            target_net.load_state_dict(policy_net.state_dict())

        # plot data
        if steps_done % RUN_TEST == 0:
            reward_test, actions_test = test(envTest, resize, 10, policy_net, device, actions_offset, False)
            writer.add_scalar('Mean Test Reward', reward_test.mean(), steps_done)
            writer.add_scalar('Std Test Reward', reward_test.std(), steps_done)
            writer.add_scalar('Mean Test Actions', actions_test.mean(), steps_done)
            writer.add_scalar('Std Test Actions', actions_test.std(), steps_done)

    # After Episode
    writer.add_scalar('Training Loss', loss, i_episode)
    writer.add_scalar('Sum Training Actions', actions.sum(), i_episode)
    writer.add_scalar('Total Training Reward', total_reward, i_episode)
    writer.add_scalar('Training Loss Steps', loss, steps_done)
    writer.add_scalar('Sum Training Actions Steps', actions.sum(), steps_done)
    writer.add_scalar('Total Training Reward Steps', total_reward, steps_done)

    if i_episode % 250 == 0:
        print("Epoch: ", i_episode, " - Total reward: ", total_reward, "Episode duration: ", episode_durations[-1],
              "Actions: ", actions, "Threshold: ", threshold)
        torch.save(policy_net.state_dict(), path)
        torch.save(policy_net.state_dict(), path.replace(".pt", f"_{i_episode}_{steps_done}.pt"))
        print("Model Saved %d" % (i_episode))

    del state_cuda
    torch.cuda.empty_cache()

torch.save(policy_net.state_dict(), path)
print('Complete')
plot_durations(episode_durations)
env.close()

