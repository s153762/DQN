import math
import random
import torch

def select_action(steps_done, episodes_done, state, n_actions, EPS_END, EPS_START, EPS_DECAY, policy_net, device):
    sample = random.random()

    if episodes_done < 400:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    else:
        eps_threshold = 0

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return steps_done, policy_net(state.to(device)).max(1)[1].view(1, 1), eps_threshold
    else:
        return steps_done, torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold

