# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from unittest import TestCase
from data.env import Env
from models.agent import Agent
from models.trainer_priority import Trainer_priority
from keras.optimizers import Adam


class TestTrainerPriority(TestCase):
    def test_train(self):
        env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        agent = Agent(input_data_shape=(10,))
        mount_agent = Agent(actions=10, input_data_shape=(10,))
        print(len(env.fx_time_data_buy))
        trainer = Trainer_priority(env, agent, mount_agent,
                                   data_end_index=len(env.fx_time_data_buy) - 2)
        trainer.train()
