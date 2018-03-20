import os
import logging
import tensorflow as tf
import pandas as pd

from huaytools.tf_.utils import DataGenerator
from huaytools.tf_.example.example_model import ExampleModel
from huaytools.tf_.example.example_trainer import ExampleTrainer

import huaytools as hy

from huaytools.tf_ import BaseConfig, BaseSummary

logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = r"D:\OneDrive\workspace\proj\huaytools\huaytools\tf_template\data\iris\iris_training.csv"
    test_path = r"D:\OneDrive\workspace\proj\huaytools\huaytools\tf_template\data\iris\iris_test.csv"

    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def main():
    (train_x, train_y), (test_x, test_y) = load_data()

    config = None

    try:
        # 使用命令行+配置文件
        # args = get_args()
        # config = process_config(args.config)

        # 使用配置文件
        # config_path = r"D:\OneDrive\workspace\proj\huaytools\huaytools\tf\example\example_config.json"
        # config = get_config(config_path)

        # 自定义
        config = BaseConfig('example', [784])
        config.summary_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "summary/"))
        config.ckpt_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "checkpoint/"))
    except:
        print("missing or invalid arguments")
        exit(0)

    sess = tf.Session()

    model = ExampleModel(config)

    # load model if exist
    model.load(sess, config.ckpt_dir)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    # logger = Logger(sess, config)
    logger = BaseSummary(sess, config)

    # create trainer and path all previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
