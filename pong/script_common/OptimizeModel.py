import torch
import torch.nn.functional as F
from torch import nn

def sample_memory(memory, BATCH_SIZE, Transition, device, non_blocking=False):
    # sample replay buffer
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # convert to tensors and create batches
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device, non_blocking=non_blocking)
    state_batch = torch.cat(batch.state).to(device, non_blocking=non_blocking)
    action_batch = torch.cat(batch.action).to(device, non_blocking=non_blocking)
    reward_batch = torch.cat(batch.reward).to(device, non_blocking=non_blocking)

    return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states

def optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer, batch=None):
    # Prepare networks
    policy_net.train()
    target_net.eval()

    if batch is None:
        batch = sample_memory(memory, BATCH_SIZE, Transition, device)

    state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = batch
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Compute expected state action value
    argmax_next_state_values = policy_net(non_final_next_states).argmax(1).detach().unsqueeze(1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, argmax_next_state_values).detach().squeeze()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.)
    optimizer.step()
    return loss.clone()


