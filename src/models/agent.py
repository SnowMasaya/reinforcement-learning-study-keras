# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np


class Agent(object):

    def __init__(self, actions: int=3, input_data_shape: int=(2,)):
        self.input_shape = input_data_shape
        self.num_actions = actions
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=input_data_shape))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        self.model = model

    def evaluate(self, state, model=None):
        _model = model if model else self.model
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        return _model.predict(_state)[0]

    def act(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            a = np.random.randint(low=0, high=self.num_actions, size=1)[0]
        else:
            q = self.evaluate(state)
            a = np.argmax(q)
        return a
