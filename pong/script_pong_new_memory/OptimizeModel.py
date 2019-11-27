import torch
import torch.nn.functional as F
from ReplayMemory import sample_memory
from torch import nn


def optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer, batch=None):
    """
    Optimize policy following the DQN objective or Double DQN objective.
    :param policy_net: policy model
    :param target_net: target policy model updated every
    :param memory: Replay memory
    :param batch: batch to use (sample memory if None)
    :param use_double_dqn: use double DQN
    :return: None
    """

    policy_net.train()
    target_net.eval()

    if len(memory) < BATCH_SIZE:
        return

    if batch is None:
        batch = sample_memory(memory, BATCH_SIZE, Transition, device)

    state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = batch

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Compute expected state action value
    argmax_next_state_values = policy_net(non_final_next_states).argmax(1).detach().unsqueeze(1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1,argmax_next_state_values).detach().squeeze()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    # compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.)
    optimizer.step()
    return loss


