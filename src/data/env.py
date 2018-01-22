# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import pickle


class Env(object):

    def __init__(self, balance: int=0, FX_DATA_FILE: str="../data/raw/FX_Demo/USD_JPY_S5.pickle"):
        self.balance = balance
        self.initial_balance = balance
        self.stock_balance = 0
        with open(FX_DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        self.fx_time_data_buy = data['highAsk']
        self.fx_time_data_sell = data['highBid']
        self.closeAsk_data = data['closeAsk']
        self.closeBid_data = data['closeBid']
        self.lowAsk_data = data['lowAsk']
        self.lowBid_data = data['lowBid']
        self.openAsk_data = data['openAsk']
        self.openBid_data = data['openBid']
        self.state = 0

    def __sell(self, mount: int=0):
        state_action = {}
        if mount * self.fx_time_data_sell[self.state] > self.stock_balance:
            state_action['fail'] = 0
            return state_action
        # print("sell {}".format(mount * self.fx_time_data_sell[self.state]))
        sell_price = mount * self.fx_time_data_sell[self.state]
        self.stock_balance = self.stock_balance - sell_price
        self.balance = self.balance + sell_price
        profit = self.balance + self.stock_balance
        self.state += 1
        state_action['success'] = profit
        return state_action

    def __buy(self, mount: int=0):
        state_action = {}
        if mount * self.fx_time_data_buy[self.state] > self.balance:
            state_action['fail'] = 0
            return state_action
        # print("buy {}".format(mount * self.fx_time_data_buy[self.state]))
        buy_price = mount * self.fx_time_data_buy[self.state]
        self.stock_balance = self.stock_balance + buy_price
        self.balance = self.balance - buy_price
        profit = self.balance + self.stock_balance
        self.state += 1
        state_action['success'] = profit
        return state_action

    def __stay(self):
        state_action = {}
        profit = 0
        self.state += 1
        state_action['success'] = profit
        return state_action

    def step(self, action: str='stay', mount: int=0):
        if action == 'stay':
            return self.__stay()
        if action == 'buy':
            return self.__buy(mount=mount)
        if action == 'sell':
            return self.__sell(mount=mount)

    def reset(self):
        self.balance = self.initial_balance
        self.stock_balance = 0
        self.state = 0
