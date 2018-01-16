# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from keras.optimizers import Adam
from keras.models import clone_model
import os
from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from collections import deque


class Trainer(object):

    def __init__(self, env, agent, mount_agent, optimizer=Adam(), model_dir="", data_end_index: int=98):
        self.env = env
        self.agent = agent
        self.mount_agent = mount_agent
        self.experience = []
        self._target_model = clone_model(self.agent.model)
        self._target_mount_model = clone_model(self.mount_agent.model)
        self.moder_dir = model_dir
        if not self.moder_dir:
            self.model_dir = os.path.join(os.path.dirname(__file__), "model")
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)
        self.agent.model.compile(optimizer=optimizer, loss='mse')
        self.mount_agent.model.compile(optimizer=optimizer, loss='mse')
        self.callback = TensorBoard(self.model_dir)
        self.callback.set_model(self.agent.model)
        self.mount_base = 100
        self.data_end_index = data_end_index
        self.name_action = {0: "buy", 1: "sell", 2: "stay"}

    def get_batch(self, batch_size: int=32, gamma=0.99, agent=None, _target_model=None):
        batch_indices = np.random.randint(low=0,
                                          high=len(self.experience),
                                          size=batch_size)
        X = np.zeros((batch_size, + agent.input_shape[0]))
        y = np.zeros((batch_size, + agent.num_actions))
        for i, b_i in enumerate(batch_indices):
            s, a, r, next_s, game_over = self.experience[b_i]
            X[i] = s
            y[i] = agent.evaluate(s)
            Q_sa = np.max(self.agent.evaluate(next_s, model=_target_model))
            if game_over:
                y[i, a] = r
            else:
                y[i, a] = r + gamma * Q_sa
        return X, y

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()

    def train(self,
              gamma: float=0.99,
              initial_epsilon: float=0.1,
              final_epsilon: float=0.0001,
              memory_size: int=50000,
              observation_epochs: int=100,
              train_epochs: int=2000,
              batch_size: int=32,
              ):
        self.experience = deque(maxlen=memory_size)
        epochs = observation_epochs + train_epochs
        epsilon = initial_epsilon
        model_path = os.path.join(self.model_dir, "agent_network.h5")
        fmt = "Epoch {:04d}/{:d} | Loss {:.5f} | Score: {} e={:.4f} train={}"

        for e in range(epochs):
            loss = 0.0
            rewards = []
            self.env.reset()
            state = (self.env.balance,
                     self.env.stock_balance,
                     self.env.fx_time_data_buy[self.env.state],
                     self.env.fx_time_data_sell[self.env.state],
                     )
            game_over = False
            is_training = True if e > observation_epochs else False

            while not game_over:
                if not is_training:
                    action = self.agent.act(state, epsilon=1)
                    mount = self.mount_agent.act(state, epsilon=1) + 1
                else:
                    action = self.agent.act(state, epsilon)
                    mount = self.mount_agent.act(state, epsilon=1) + 1

                reward = self.env.step(action=self.name_action[action],
                                       mount=mount * self.mount_base)
                if "success" in reward:
                    reward = reward["success"]
                elif "fail" in reward:
                    # print("******** fail process *************")
                    reward = reward["fail"]

                next_state = (self.env.balance,
                              self.env.stock_balance,
                              self.env.fx_time_data_buy[self.env.state],
                              self.env.fx_time_data_sell[self.env.state],
                              )
                if self.env.balance == 0 or self.env.state > self.data_end_index:
                    game_over = True
                self.experience.append(
                    (state, action, reward, next_state, game_over))

                rewards.append(reward)
                # print("mount {}".format(mount))
                # print("reward {}".format(reward))

                if is_training:
                    X, y = self.get_batch(batch_size, gamma,
                                          agent=self.agent,
                                          _target_model=self._target_model)
                    loss += self.agent.model.train_on_batch(X, y)
                    X, y = self.get_batch(batch_size, gamma,
                                          agent=self.mount_agent,
                                          _target_model=self._target_mount_model)
                    loss += self.mount_agent.model.train_on_batch(X, y)

                state = next_state

            loss= loss / len(rewards)
            score = sum(rewards)

            if is_training:
                self.write_log(e - observation_epochs, loss, score)
                self._target_model.set_weights(self.agent.model.get_weights())
                self._target_mount_model.set_weights(self.mount_agent.model.get_weights())

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / epochs

            print(fmt.format(e + 1, epochs, loss, score, epsilon, is_training))
            print("balance {}, stock_balance {} total_balance {}".
                  format(self.env.balance, self.env.stock_balance, self.env.balance + self.env.stock_balance))

            if e % 100 == 0:
                self.agent.model.save(model_path, overwrite=True)

        self.agent.model.save(model_path, overwrite=True)
