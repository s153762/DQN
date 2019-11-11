import random

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.m = [] # memory
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.m) < self.capacity:
            self.m.append(None)
        self.m[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.m, batch_size)

    def __len__(self):
        return len(self.m)