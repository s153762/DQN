import torch
import torch.nn.functional as F

def optimize_model(memory, BATCH_SIZE, Transition, device, policy_net, target_net, GAMMA, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch)  # .gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros((BATCH_SIZE, 3), device=device)
    next_state_values[non_final_mask.bool()] = target_net(non_final_next_states)

    next_state_values = next_state_values.max(1)[0].detach()

    q_targets = state_action_values.clone()
    # Compute the expected Q values
    q_targets[range(BATCH_SIZE), action_batch - 1] = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss

    loss = F.smooth_l1_loss(state_action_values, q_targets)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
