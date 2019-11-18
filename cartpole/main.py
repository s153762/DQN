import gym
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.optim as optim
import torchvision.transforms as T

from cartpole.ReplayMemory import ReplayMemory # Get ReplayMemory
from cartpole.DQN import DQN # Get Network
from cartpole.get_screen import get_screen
from cartpole.select_action import select_action
from cartpole.plot_durations import plot_durations
from cartpole.optimize_model import optimize_model

env = gym.make('CartPole-v1').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    print("Is using ipython display")

plt.ion()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Is using %s as device" % device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(32, interpolation=Image.CUBIC),
                    T.ToTensor()])

env.reset()
plt.figure()
plt.imshow(get_screen(env, resize, device).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.axis('off')
plt.savefig("plt/ExampleExtracted Screen")
plt.show()


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(env, resize, device)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000, Transition)

episode_durations = []

#%%

num_episodes = 20

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env, resize, device)
    current_screen = get_screen(env, resize, device)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state, EPS_START, EPS_END, EPS_DECAY, policy_net, n_actions, device)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env, resize, device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(memory, BATCH_SIZE, Transition, policy_net, target_net, GAMMA, optimizer, device)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations, is_ipython, display)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

plt.ioff()
env.render()
env.close()
print('Complete')
