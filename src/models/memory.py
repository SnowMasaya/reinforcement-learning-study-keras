# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from collections import deque
import numpy as np


class Memory(object):

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


class MemoryTDerror(Memory):

    def __init__(self, max_size=1000):
        super().__init__(max_size)

    def get_TDerror(self, memory, gamma, mainQN, targetQN):
        (state, action, reward, next_state) = memory.buffer[memory.len() - 1]
        next_state = np.array(next_state)
        state = np.array(state)
        next_action = np.argmax(mainQN.model.predict(next_state)[0])
        target = reward + gamma * targetQN.model.predict(next_state)[0][next_action]
        TDerror = target - targetQN.model.predict(state)[0][action]
        return TDerror

    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state) = memory.buffer[i]
            next_action = np.argmax(mainQN.model.predict(next_state)[0])
            target = reward + gamma * targetQN.model.predict(next_state)[0][next_action]
            TDerror = target - targetQN.model.predict(state)[0][action]
            self.buffer[i] = TDerror

    def get_sum_absolute_TDerrpr(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001

        return sum_absolute_TDerror