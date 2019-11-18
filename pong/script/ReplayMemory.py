import random

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)