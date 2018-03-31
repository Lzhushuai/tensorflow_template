"""
Show how to use `tf.feature_columns`
"""
import logging
import tensorflow as tf

from tensorflow_template import BaseModel

logger = logging.getLogger(__name__)


class IrisModel(BaseModel):
    def _init_graph(self):
        self.features = tf.placeholder(tf.float32, [None] + self.config.n_feature, 'features')
        self.labels = tf.placeholder(tf.int32, [None], 'labels')

        net = self.features  # input_layer
        for units in self.config.n_units:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            # net = tf.layers.Dense(units=self.config.n_units[0], activation=tf.nn.relu)(net)

        self.logits = tf.layers.dense(net, self.config.n_class, activation=None)
        self.prediction = tf.argmax(self.logits, axis=1)

        self.accuracy, self.update_op = tf.metrics.accuracy(labels=self.labels,
                                                            predictions=self.prediction,
                                                            name='acc_op')

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self._global_step)

    def train(self, dataset, *args, **kwargs):
        pass

    def evaluate(self, dataset, *args, **kwargs):
        pass

    def predict(self, dataset, *args, **kwargs):
        pass