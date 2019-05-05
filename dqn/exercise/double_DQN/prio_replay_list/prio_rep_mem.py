from sortedcontainers import SortedList
from collections import namedtuple
import random
import numpy as np


class SuperTuple:

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    d = {'e': None, 'p': None, 'p_prob': None}

    def __init__(self, state, action, reward, next_state, done, priority, priority_prob):
        t = self.experience(state, action, reward, next_state, done)
        self.d = {'t': t, 'p': priority, 'p_prob': priority_prob}

    def __lt__(self, o):
        return self.d['p_prob'] > o.d['p_prob']

    def __gt__(self, o):
        return self.d['p_prob'] < o.d['p_prob']

    def __le__(self, o):
        return self.d['p_prob'] >= o.d['p_prob']

    def __ge__(self, o):
        return self.d['p_prob'] <= o.d['p_prob']

    def __repr__(self):
        return str(self.d)


class MySortedList:

    buf_len = 1000
    replay_alpha = 0.6
    replay_beta = 0.4
    sl = SortedList()
    num_items = 0
    sum_priorities = 0
    sum_priority_probs = 0
    bins_num = 64
    _delta = 0
    max_prio = 0
    min_prio = 0

    def __init__(self, buf_len=1000, bins_num=64, replay_alpha=0.6, replay_beta = 0.4, rep_beta_inc = 0.001):
        self.buf_len = buf_len
        self.bins_num = bins_num
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.beta_increment_per_sampling = rep_beta_inc

    def add(self, s):
        if self.num_items < self.buf_len:
            self.num_items += 1
            self.sum_priorities += s.d['p']
            self.sum_priority_probs += s.d['p_prob']
        else:
            expelled = self.sl.pop()
            self.sum_priorities -= expelled.d['p']
            self.sum_priority_probs -= expelled.d['p_prob']
        self.sl.add(s)

    def pop(self, idx):
        expelled = self.sl.pop(idx)
        self.num_items -= 1
        self.sum_priorities -= expelled.d['p']
        self.sum_priority_probs -= expelled.d['p_prob']
        return expelled

    def sample(self, bins_num=64):
        self.bins_num = bins_num
        self.max_prio = self.sl[0].d['p_prob']
        self.min_prio = self.sl[-1].d['p_prob']
        self._delta = (self.max_prio - self.min_prio) / self.bins_num
        exps = []
        idxs = []
        idx0 = len(self.sl)
        for i in range (0, self.bins_num):
            idx1 = idx0
            if i != self.bins_num - 1:
                temp = self.min_prio + (self._delta * (i+1))
                idx0 = self.sl.bisect_right(SuperTuple(0, 0, 0, 0, 0, 0, temp))
            else:
                idx0 = 0

            #Handle case in which both indexes refer to the same element
            if idx0 == idx1:
                #print("equals!")
                idx1 += 1
            # Handle equiprobable case (i.e. replay_alpha = 0)
            if self.replay_alpha == 0:
                idx0 = 0
                idx1 = len(self.sl)
            
            index = min(round(random.uniform(idx0, idx1)), len(self.sl)-1)
            #print("idx0: " + str(idx0) + " idx1: " + str(idx1) + " length: " + str(len(self.sl)) + " index: " + str(index))
            idxs.append(index)
            exps.append(self.sl[index])
            #exps.append(random.sample(self.sl[idx0:idx1], k=1))
            #print("success")
        probs = [x.d['p_prob'] for x in exps]
        probs = np.asarray(probs) / self.sum_priority_probs
        self.replay_beta = np.min([1., self.replay_beta + self.beta_increment_per_sampling])
        ws = (probs * self.buf_len) ** (-self.replay_beta)
        ws /= np.max(ws)
        return exps, idxs, ws, probs

    def update_batch(self, idxs, errors):
        for i in range(0, len(idxs)):
            self._update(idxs[i], errors[i])

    def _update(self, idx, error):
        expelled = self.pop(idx)
        self.add(SuperTuple(expelled.d['t'].state,
                            expelled.d['t'].action,
                            expelled.d['t'].reward,
                            expelled.d['t'].next_state,
                            expelled.d['t'].done,
                            error,
                            error**self.replay_alpha))

    def __repr__(self):
        return str(self.sl)

    def __len__(self):
        return len(self.sl)


'''

ra = 1
e1 = SuperTuple(1, 1, 1, 2, 0, 3, 3**ra)
e2 = SuperTuple(2, 1, 1, 3, 0, 1, 1**ra)
e3 = SuperTuple(3, 1, 1, 4, 0, 7, 7**ra)
e4 = SuperTuple(4, 1, 1, 5, 0, 2, 2**ra)
e5 = SuperTuple(5, 1, 1, 6, 0, 9, 9**ra)
e6 = SuperTuple(6, 1, 1, 7, 0, 4, 4**ra)

e7 = SuperTuple(7, 1, 1, 8, 0, 5, 5**ra)
e8 = SuperTuple(8, 1, 1, 9, 0, 4, 4**ra)
e9 = SuperTuple(9, 1, 1, 10, 0, 20, 20**ra)
e10 = SuperTuple(10, 1, 1, 11, 0, 14, 14**ra)

msl = MySortedList(buf_len=1000, bins_num=3, replay_alpha=ra)
msl.add(e1)
msl.add(e2)
msl.add(e3)
msl.add(e4)
msl.add(e5)
msl.add(e6)
msl.add(e7)
msl.add(e8)
msl.add(e9)
msl.add(e10)

print(msl)

sample, idxs = msl.sample(bins_num=4)
print(sample)

'''