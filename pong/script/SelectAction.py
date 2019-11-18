import math
import random
import torch

steps_done = 0

def select_action(state, EPS_END, EPS_START, EPS_DECAY, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-0.001 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) + 1, eps_threshold
    else:
        return torch.tensor([[random.randrange(3) + 1]], device=device, dtype=torch.long), eps_threshold
