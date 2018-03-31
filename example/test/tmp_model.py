import huaytools as hy
import logging

import numpy as np
import tensorflow as tf
from tensorflow_template.base.base_model import BasicModel, BaseConfig
from tensorflow_template.utils.tools import set_logging_basic_config

# logger = logging.getLogger(__name__)
set_logging_basic_config(level=logging.DEBUG)


# logger = hy.get_logger(__name__, level=logging.DEBUG)


class TmpModel(BasicModel):

    # def _init_placeholder(self):
    #     with self.graph.as_default():
    #         self.features_ph = tf.placeholder(tf.float32, [None, 10], 'x')
    #         self.labels_ph = tf.placeholder(tf.int32, [None], 'y')

    def _init_model(self, features, labels):

        net = tf.layers.dense(features, units=32, activation=tf.nn.relu)

        self._logits = tf.layers.dense(net, self.config.n_class, activation=None)

        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)

        self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._config.learning_rate)
        self._train_op = self._optimizer.minimize(self._loss, global_step=tf.train.get_global_step())

    # def _train_model(self, sess, features, labels):
    #     while True:
    #         try:
    #             loss_val, _, step = sess.run([self._loss, self._train_op, self.global_step])
    #             # logger.info("Step {} Loss {:.3}".format(step, loss_val))
    #         except tf.errors.OutOfRangeError:
    #             break
    #
    #         self.save(sess)


def get_config():
    config = BaseConfig('Tmp')

    config.ckpt_dir = './log/ckpt'
    config.n_class = 5
    config.n_batch = 100
    config.n_epoch = 10

    return config


def get_dataset():
    n = 100000
    features = np.random.random((n, 10))
    labels = np.array([0, 1, 2, 3, 4] * (n//5))

    return tf.data.Dataset.from_tensor_slices((features, labels))


def tmp_model(config, dataset):
    """The whole process if not use class"""
    dataset = dataset.shuffle(config.buffer_size).batch(config.n_batch).repeat(100)
    ds_iter = dataset.make_one_shot_iterator()

    global_step = tf.Variable(0, trainable=False)

    features, labels = ds_iter.get_next()
    x_ = tf.placeholder(tf.float32, [None, 10], name='x')
    y_ = tf.placeholder(tf.int32, [None], name='y')

    def model_(x, y):
        net = tf.layers.dense(x, units=32, activation=tf.nn.relu)
        logits = tf.layers.dense(net, 2, activation=None)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return loss, train_op

    loss, train_op = model_(x_, y_)
    print(loss)

    # with tf.Session() as sess:
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     init_op.run()
    #
    #     while True:
    #         try:
    #             loss_val, _, step = sess.run([loss, train_op, global_step])
    #             logger.info("Step {} Loss {:.3}".format(step, loss_val))
    #         except tf.errors.OutOfRangeError:
    #             break


def how_saver_work():
    g = tf.Graph()

    with g.as_default():
        a = tf.Variable(1, name='a')
        b = tf.Variable(2, name='b')

        print(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
        saver = tf.train.Saver()

        c = tf.Variable(3, name='c')
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            print(a.eval())
            print(b.eval())
            print(c.eval())

            save_path = saver.save(sess, "./tmp/model.ckpt")

    with tf.Session(graph=g) as sess:
        print(g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.restore(sess, save_path)

        print(a.eval())
        print(b.eval())
        print(c.eval())  # err: uninitialized


if __name__ == '__main__':
    """"""
    # logger.setLevel(logging.DEBUG)
    #
    config = get_config()

    dataset = get_dataset()

    model = TmpModel(config)

    # with tf.Session(graph=model.graph) as sess:
    model.train(dataset)

        # model.load(sess)

    # tmp_model(config, dataset)

    # how_saver_work()

    # from tensorflow.python.tools import inspect_checkpoint as chkp
    #
    # latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir)
    #
    # # print all tensors in checkpoint file
    # chkp.print_tensors_in_checkpoint_file(file_name=latest_ckpt,
    #                                       tensor_name='', all_tensors=True, all_tensor_names=True)
