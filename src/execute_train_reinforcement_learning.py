# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from data.env import Env
from models.agent import Agent
from models.trainer import Trainer
from keras.optimizers import Adam
import argparse


def main():

    parser = argparse.ArgumentParser(description='Execute train reinforcement learning.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="../data/raw/FX_Demo/sample10000_USD_JPY_S5.pickle",
                        help='an integer for the accumulator')

    args = parser.parse_args()
    print(args.dataset_name)
    env = Env(balance=250000, FX_DATA_FILE=args.dataset_name)
    agent = Agent(input_data_shape=(10,))
    mount_agent = Agent(actions=10, input_data_shape=(10,))
    trainer = Trainer(env, agent, mount_agent, Adam(lr=1e-6), data_end_index=len(env.fx_time_data_buy) - 2)
    trainer.train()


if __name__ == "__main__":
    main()