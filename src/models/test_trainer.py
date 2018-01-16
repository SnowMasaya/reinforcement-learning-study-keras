# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from unittest import TestCase
from data.env import Env
from models.agent import Agent
from models.trainer import Trainer
from keras.optimizers import Adam


class TestTrainer(TestCase):
    def test_train(self):
        env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        agent = Agent(input_data_shape=(4,))
        mount_agent = Agent(actions=10, input_data_shape=(4,))
        print(len(env.fx_time_data_buy))
        trainer = Trainer(env, agent, mount_agent, Adam(lr=1e-6), data_end_index=len(env.fx_time_data_buy) - 2)
        trainer.train()
