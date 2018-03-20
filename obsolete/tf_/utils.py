import os
import json
import argparse

import numpy as np
from bunch import Bunch


def get_config(json_file):
    """从 json 文件读取所有参数
    json 文件中的结构应该是一个 字典，如果不是字典，需要重写该方法
        示例
        {
          "name": "example",
          "n_feature": [784]
          "n_batch": 16,
          "n_epoch": 10,
          "n_step": 10,
          "learning_rate": 0.001,
        }


        如果是从命令行运行程序，推荐使用该方法，配合 get_config_args()
        如果是在 IDE 中调试，建议使用 ``base.BaseConfig``
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config


def get_config_args():
    """如果从命令行执行，只需要一个 config.json 文件的路径参数

        >>> args = get_config_args()
        >>> config = get_config(args.config)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file path')
    args = parser.parse_args()

    return args


class DataGenerator():
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
