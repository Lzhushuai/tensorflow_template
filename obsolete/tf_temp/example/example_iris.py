"""
example for iris data with a two-hidden-layers net
"""

import os
import logging
import tensorflow as tf
import huaytools as hy

from huaytools.tf_temp.utils.Config import Config
from huaytools.tf_temp.base.base_model import BaseModel
from huaytools.tf_temp.utils.data_iris import *

logger = logging.getLogger(__name__)


class IrisModel(BaseModel):
    """"""

    def __init__(self, config, feature_columns):
        super(IrisModel, self).__init__(config)

        self.fcs = feature_columns
        self.mode = tf.estimator.ModeKeys.TRAIN

    def build_graph(self):
        if self.graph is None:
            self.graph = tf.get_default_graph()

        with self.graph.as_default():
            self._init_model()
            self._init_loss()
            self._init_train_op()

    def _init_input(self):
        pass

    def _init_model(self, *args, **kwargs):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.features = self.ds_iter.get_next()
        else:
            self.features, self.labels = self.ds_iter.get_next()

        self.net = tf.feature_column.input_layer(self.features, self.fcs)

        for units in self.config.hidden_units:
            self.net = tf.layers.dense(self.net, units=units, activation=tf.nn.relu)

        self.logits = tf.layers.dense(self.net, units=self.config.n_class, activation=None)

        self.predicted_classes = tf.argmax(self.logits, axis=1)

    def train(self, sess, ds_iter, *args, **kwargs):
        self.set_ds_iter(ds_iter)
        # self.features, self.labels = self.ds_iter.get_next()
        self.build_graph()
        tf.global_variables_initializer().run()

        while True:
            try:
                _, loss_value = sess.run([self.train_op, self.loss])
                print(loss_value)
            except tf.errors.OutOfRangeError:
                break

    def evaluate(self, sess, ds_iter, *args, **kwargs):
        pass

    def predict(self, sess, ds_iter, *args, **kwargs):
        self.mode = tf.estimator.ModeKeys.PREDICT
        self.set_ds_iter(ds_iter)

        if self.graph is None:
            self._init_model()

        pred_ret = []
        while True:
            try:
                ret = sess.run(self.predicted_classes)
                pred_ret += ret
            except tf.errors.OutOfRangeError:
                break

        return pred_ret

    def _init_loss(self, *args, **kwargs):
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

    def _init_train_op(self, *args, **kwargs):
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


if __name__ == '__main__':
    config = Config('iris')
    config.hidden_units = [10, 10]
    config.n_class = 3
    config.ckpt_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "checkpoint/"))

    train_ds_iter = get_train_ds_iter()
    print(train_ds_iter.output_shapes)

    pred_ds_iter = get_pred_ds_iter()
    print(pred_ds_iter.output_shapes)

    feature_columns = [
        tf.feature_column.numeric_column('SepalLength'),
        tf.feature_column.numeric_column('SepalWidth'),
        tf.feature_column.numeric_column('PetalLength'),
        tf.feature_column.numeric_column('PetalWidth')
    ]

    model = IrisModel(config, feature_columns)

    features, labels = train_ds_iter.get_next()

    with tf.Session() as sess:

        model.load(sess)

        model.train(sess, train_ds_iter)

        model.save(sess)
