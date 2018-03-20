import os
import huaytools as hy
import tensorflow as tf
from example.example_model import ExampleModel
from base.base_config import Config
from utils.data_iris_fc import *


def _parse_line(line):
    return tf.decode_csv(line, record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])


if __name__ == '__main__':
    config = Config('ex', [4], 3)
    config.ckpt_dir = "D:/Tmp/log/example_ckpt/"
    config.n_feature = [4]
    config.n_units = [10, 10]
    config.n_class = 3

    fcs = [
        tf.feature_column.numeric_column('SepalLength'),
        tf.feature_column.numeric_column('SepalWidth'),
        tf.feature_column.numeric_column('PetalLength'),
        tf.feature_column.numeric_column('PetalWidth')
    ]

    dataset = tf.data.TextLineDataset(
        r"D:\OneDrive\workspace\py\DL\tensorflow_template\data\iris\iris_training.csv").skip(1).batch(12).map(_parse_line)


