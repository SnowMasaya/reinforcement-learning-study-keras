# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from models.memory import Memory


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class QNetWork(object):
    def __init__(self,
                 learning_rate=0.01,
                 state_size=10,
                 actions_size=3,
                 hidden_size=10):
        self.state_size = state_size
        self.actions_size = actions_size
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(actions_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.actions_size))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target
            self.model.train_on_batch(inputs, targets)

    def prioritized_experience_replay(self, memory, batch_size, gamma,
                                      targetQN, memory_TDerror):
        sum_absoluate_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absoluate_TDerror, batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        batch_memory = Memory(max_size=batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) + 0.0001
                idx += 1
            batch_memory.add(memory.buffer[idx])

        inputs = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.actions_size))
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(batch_memory.buffer):
            inputs[i:i + 1] = state_b
            targets = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target
            self.model.train_on_batch(inputs, targets)
