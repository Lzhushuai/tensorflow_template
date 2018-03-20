import os
import logging
import tensorflow as tf
import huaytools as hy

from huaytools.tf_temp.utils.Config import Config
from huaytools.tf_temp.base.base_model import BaseModel

logger = logging.getLogger(__name__)


class ExampleModel(BaseModel):
    """"""

    def __init__(self, config):
        super(ExampleModel, self).__init__(config)

    def build_graph(self):
        if self.graph is None:
            self.graph = tf.get_default_graph()

        with self.graph.as_default():
            self._init_input()
            self._init_model()
            self._init_loss()
            self._init_train_op()

    def _init_input(self):
        self.x, self.y_true = self.ds_iter.get_next()

    def _init_model(self, *args, **kwargs):
        self.model = tf.layers.Dense(units=1)
        self.y_pred = self.model(self.x)

    def train(self, sess, ds_iter, *args, **kwargs):
        self.set_ds_iter(ds_iter)
        self.build_graph()
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                _, loss_value = sess.run([self.train_op, self.loss])
                # print(loss_value)
            except tf.errors.OutOfRangeError:
                break

    def evaluate(self, sess, ds_iter, *args, **kwargs):
        pass

    def predict(self, sess, ds_iter, *args, **kwargs):
        x, y = ds_iter.get_next()
        # print(sess.run(x))
        while True:
            try:
                yield sess.run(self.model(x))
                # print(ret)
            except tf.errors.OutOfRangeError:
                break

    def _init_loss(self, *args, **kwargs):
        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_pred)

    def _init_train_op(self, *args, **kwargs):
        with tf.name_scope('train_op'):
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
    #
    # def init_global_step(self, *args, **kwargs):
    #     with tf.variable_scope("global_step"):
    #         self.global_step = tf.Variable(0, trainable=False, name='global_step')
    #
    # def init_saver(self, max_to_keep=5):
    #     self.saver = tf.train.Saver(max_to_keep=max_to_keep)


if __name__ == '__main__':
    config = Config('example')
    config.summary_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "summary/"))
    config.ckpt_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "checkpoint/"))

    # the dataset
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y_true))

    # model
    model = ExampleModel(config)
    # model.init_ds_iter(ds.make_one_shot_iterator())
    # model.build()

    with tf.Session() as sess:
        model.load(sess)

        ds_iter = ds.repeat(100).batch(100).make_one_shot_iterator()
        model.train(sess, ds_iter)
        ds_iter = ds.repeat(2).batch(2).make_one_shot_iterator()
        ret = model.predict(sess, ds_iter)
        for i in ret:
            print(i)

        model.save(sess)


