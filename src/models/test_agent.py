# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.env import Env
from models.agent import Agent
from keras.optimizers import Adam


class TestAgent(TestCase):
    def test_evaluate(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        self.agent = Agent()
        self.agent.model.compile(optimizer=Adam(), loss="mse")
        state = (self.env.balance, self.env.stock_balance)
        y = self.agent.evaluate(state=state)
        assert y.shape == (3,)
        assert any(y) is True

    def test_act(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        self.agent = Agent()
        self.agent.model.compile(optimizer=Adam(), loss="mse")
        state = (self.env.balance, self.env.stock_balance)
        action = self.agent.act(state, epsilon=0.1)
        action_state = False
        if action == 0 or action == 1 or action == 2:
            action_state = True
        assert action_state is True


