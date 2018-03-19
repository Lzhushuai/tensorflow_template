import os
import logging
import tensorflow as tf
import huaytools as hy

from utils.Config import Config
from base.base_model import BaseModel

logger = logging.getLogger(__name__)


class ExampleModel(BaseModel):
    """"""
    def __init__(self, config, feature_columns=None, graph=None):
        super(ExampleModel, self).__init__(config, feature_columns, graph)

    def _init_graph(self, features, labels=None):
        with self.graph.as_default():
            features, labels = self.ds_iter.get_next()

            net = tf.feature_column.input_layer(features, self.feature_columns)

            for units in self.config.n_units:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

            self.logits = tf.layers.dense(net, self.config.n_class, activation=None)
            self.prediction = tf.argmax(self.logits, axis=1)

            if labels is not None:  # train or evaluate
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)

                optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            self._init_saver()

    def train(self, sess, ds_iter, *args, **kwargs):
        self.set_ds_iter(ds_iter)
        feature, labels = self.ds_iter.get_next()
        self._init_graph(feature, labels)
        sess.run(tf.global_variables_initializer())

        # for _ in range(self.config.n_epoch):
        i = 1
        for _ in range(self.config.n_epoch):
            sess.run(self.ds_iter.initializer)
            while True:
                try:
                    loss_value, _ = sess.run([self.loss, self.train_op])
                    print(i, loss_value)
                    i += 1
                except tf.errors.OutOfRangeError:
                    break
            self.save(sess)

    def evaluate(self, sess, ds_iter, *args, **kwargs):
        pass

    def predict(self, sess, ds_iter, *args, **kwargs):
        pass

    def _init_saver(self, max_to_keep=5, *args, **kwargs):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=max_to_keep, *args, **kwargs)
