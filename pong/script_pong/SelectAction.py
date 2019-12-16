import math
import random
import torch

steps_done = 0

def select_action(state, n_actions, EPS_END, EPS_START, EPS_DECAY, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return steps_done, policy_net(state.to(device)).max(1)[1].view(1, 1), eps_threshold
    else:
        return steps_done, torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold

