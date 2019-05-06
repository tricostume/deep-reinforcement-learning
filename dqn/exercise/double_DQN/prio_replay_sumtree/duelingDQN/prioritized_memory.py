import random
import numpy as np
from SumTree import SumTree
from collections import namedtuple


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    max_prio = 0

    def __init__(self, capacity, replay_beta = 0.4, replay_alpha = 0.6, replay_beta_increment = 0.0001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = replay_alpha
        self.beta = replay_beta
        self.beta_increment_per_sampling = replay_beta_increment

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        if error > self.max_prio:
            self.max_prio = error
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if p == 0:
                #print("\nattempted: value " + str(s) + " at segment " + str(a) + " " + str(b) + " out of max possible " + str(self.tree.total()))
                while p == 0:
                    s = random.uniform(a, b)
                    (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_batch(self, idxs, errors):
        for i in range(0, len(idxs)):
            self._update(idxs[i], errors[i])

    def _update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return  len(self.tree)

mem = Memory(capacity = 1000)
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
e1 = experience(0, 1, 1, 1, 0)
e2 = experience(1, 1, 1, 2, 0)
e3 = experience(2, 1, 1, 3, 0)

e4 = experience(3, 1, 1, 4, 0)
e5 = experience(4, 1, 1, 5, 0)
e6 = experience(5, 1, 1, 6, 0)

e7 = experience(6, 1, 1, 7, 0)
e8 = experience(7, 1, 1, 8, 0)
e9 = experience(8, 1, 1, 9, 0)

mem.add(1, e1)
mem.add(3, e2)
mem.add(5, e3)

mem.add(6, e4)
mem.add(7, e5)
mem.add(22, e6)

mem.add(23, e7)
mem.add(24, e8)
mem.add(30, e9)

batch, idxs, is_weight = mem.sample(9)
lala = 1
