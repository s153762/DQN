#Prioritized Experience Replay
import random
from SumTree import SumTree
import torch


class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    def __init__(self, capacity, discount_factor, Transition, device):
        self.tree = SumTree(capacity)
        self.discount_factor = discount_factor
        self.transition = Transition
        self.device = device

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, *args):
        p = self._getPriority(error)
        self.tree.add(p, self.transition(*args))

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def push(self, state, action, reward, next_state, done, model, target_model):
        target = model(state.to(self.device))
        old_val = target[0][action]
        if done:
            target[0][action] = reward
        else:
            target_val = target_model(next_state.to(self.device))
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])
        self.add(error, state, action, next_state, reward)
