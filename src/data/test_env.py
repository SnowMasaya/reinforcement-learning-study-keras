# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.env import Env


class TestEnv(TestCase):
    def test_sell(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        sell_mount = self.env.step(action='sell', mount=1)
        assert sell_mount == {'fail': 0}
        self.env.step(action='buy', mount=1)
        sell_mount = self.env.step(action='sell', mount=1)
        assert sell_mount == {'success': 250000.0}
        assert self.env.stock_balance == 0.000999999999990564
        assert self.env.balance == 249999.999

    def test_buy(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        buy_mount = self.env.step(action='buy', mount=1)
        assert buy_mount == {'success': 250000.0}
        assert self.env.stock_balance == 111.332
        assert self.env.balance == 249888.668

    def test_stay(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        stay_mount = self.env.step(action='stay', mount=1)
        assert stay_mount == {'success': 0}
        assert self.env.stock_balance == 0
        assert self.env.balance == 250000

    def test_reset(self):
        self.env = Env(balance=250000, FX_DATA_FILE='../data/raw/FX_Demo/sample_USD_JPY_S5.pickle')
        self.env.step(action='buy', mount=1)
        self.env.step(action='sell', mount=1)
        self.env.reset()
        assert self.env.stock_balance == 0
        assert self.env.balance == 250000
