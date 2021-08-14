from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'action_bc, state0, action, reward, state1, terminal1')
# Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):  # maxlen即num_episodes = capacity // max_train_traj_len  #capacity=args.rmsize=6,000,000
        self.maxlen = maxlen  #即 max number of trajectories
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):  # memory.sample時從此呼叫
        if idx < 0 or idx >= self.length:
            raise KeyError(idx)
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v
        
#比較：上述RingBuffer與利用RingBuffer=deque(maxlen=capacity)的寫法？
#用ring法是append滿後，index重頭append。
#用deque法是append滿後，index最頭的那筆開始自動移除


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {'window_length': self.window_length,
                  'ignore_episode_boundaries': self.ignore_episode_boundaries,}
        return config


### 雖然有繼承Memory，但好像沒有Memory也可以運行耶？ ####
class EpisodicMemory(Memory):
    def __init__(self, capacity, max_train_traj_len, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)
        # Max number of transitions possible will be the memory capacity, could be much less
        self.max_train_traj_len = max_train_traj_len
        self.num_segmt_traj = capacity // max_train_traj_len  #capacity=args.rmsize=6,000,000
        self.memory = RingBuffer(self.num_segmt_traj)
        # self.memory = deque(maxlen=capacity)
        self.trajectory = [] # Temporal list of episode

    def append(self, action_bc, state0, action, reward, terminal, training=True):
        #先sample出一個固定maxlen的traj
        self.trajectory.append(Experience(action_bc=action_bc, state0=state0, action=action, reward=reward, state1=None, terminal1=terminal)) 
        
        # 因為self.max_train_traj_len = args.trajectory_length = 10
        # 所以整段episode會被分段存取？ 所以replay buffer裡每段episode長度只有10。
        # print('train_traj_len=', len(self.trajectory))
        if len(self.trajectory) >= self.max_train_traj_len:  
            # 用RingBuffer存trajectory一個一個存起來。
            self.memory.append(self.trajectory)
            self.trajectory = []

    # def sample(self, batch_size, maxlen=None):
    def sample(self, batch_size, maxlen=0):
        #### sample a batch of trajectories ####
        batch = [self.sample_trajectory(maxlen=maxlen) for _ in range(batch_size)]
        
        #### Truncate trajectories aligned to the minlen_traj ####
        minimum_size = min(len(trajectory) for trajectory in batch)  # find minlen_traj
        batch = [trajectory[:minimum_size] for trajectory in batch]  # truncate
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def sample_trajectory(self, maxlen=0):
        e = random.randrange(len(self.memory))
        mem = self.memory[e] #random sample出一個在memory裡index為e的trajectory
        T = len(mem)
        # print('T=',T)
        if T > 0:
            # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
            # print('maxlen=',maxlen)
            if maxlen > 0 and T > maxlen + 1:
                t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
                return mem[t:t + maxlen + 1]
            else:
                return mem

    def __len__(self):
        return sum(len(self.memory[idx]) for idx in range(len(self.memory)))

