import os
import logging
import tensorflow as tf
import tensorlayer as tl
import huaytools as hy

from base.base_config import Config
from base.base_model import BaseModel
from utils.utils import get_uninitialized_variables

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class ExampleModel(BaseModel):
    """"""

    def _init_graph(self):
        self.features = tf.placeholder(tf.float32, [None] + self.config.n_feature, 'features')
        self.labels = tf.placeholder(tf.int32, [None], 'labels')

        # net = tf.feature_column.input_layer(self.features, self.feature_columns)
        net = self.features
        for units in self.config.n_units:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        # net = tf.layers.Dense(units=self.config.n_units[0], activation=tf.nn.relu)(self.features)
        # net = tf.layers.Dense(units=self.config.n_units[0], activation=tf.nn.relu)(net)

        self.logits = tf.layers.dense(net, self.config.n_class, activation=None)
        self.prediction = tf.argmax(self.logits, axis=1)

        self.accuracy, self.update_op = tf.metrics.accuracy(labels=self.labels,
                                                            predictions=self.prediction,
                                                            name='acc_op')

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self, sess, dataset, buffer_size=1000, *args, **kwargs):
        if self.mode == self.ModeKeys.TRAIN:
            sess.run(self.init_op)

        for _ in range(self.config.n_epoch):
            ds_iter = dataset.shuffle(buffer_size).batch(self.config.n_batch).make_one_shot_iterator()
            while True:
                try:
                    features, labels = sess.run(ds_iter.get_next())
                    loss_val, _, acc_val, _ = sess.run([self.loss, self.train_op, self.accuracy, self.update_op],
                                                       feed_dict={self.features: features, self.labels: labels})
                    logger.info("Step {}: loss {}, accuracy {}".format(self.global_step.eval(sess), loss_val, acc_val))
                except tf.errors.OutOfRangeError:
                    break
            self.save(sess)

    def evaluate(self, sess, dataset, *args, **kwargs):
        self.mode = self.ModeKeys.EVAL
        ds_iter = dataset.batch(1).make_one_shot_iterator()

        while True:
            try:
                features, labels = sess.run(ds_iter.get_next())
                accuracy_val, _ = sess.run([self.accuracy, self.update_op], feed_dict={self.features: features,
                                                                                       self.labels: labels})
                print(accuracy_val)
            except tf.errors.OutOfRangeError:
                break

    def predict(self, sess, dataset, *args, **kwargs):
        pass


if __name__ == '__main__':
    config = Config('ex', [4], 3)
    config.ckpt_dir = "D:/Tmp/log/example_ckpt/"
    config.n_epoch = 50
    config.n_feature = [4]
    config.n_units = [10, 10]
    config.n_class = 3

    model = ExampleModel(config)

    from utils.data_iris import *

    ds_train = get_dataset('train')
    ds_eval = get_dataset('eval')

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        model.load(sess)
        model.train(sess, ds_train)
        print(sess.run(model.global_step))
        model.evaluate(sess, ds_eval)

    print('\n\n\n')
    # dataset = None
    #
    # with tf.Session() as sess:
    #     assert model.graph == tf.get_default_graph()
    #     assert sess.graph == tf.get_default_graph()
    #
    #     # model.load(sess)  # some error
    #
    #     model.train(sess, dataset)
    #
    #     for i, obj in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
    #         print(i, obj)
    #
    #     # f, l = ds_iter.get_next()
    #     # while True:
    #     #     try:
    #     #         ff = sess.run(f)
    #     #         print(ff)
    #     #     except tf.errors.OutOfRangeError:
    #     #         break
    #
    # from tensorflow.python.tools import inspect_checkpoint as chkp
    #
    # latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir)
    #
    # chkp.print_tensors_in_checkpoint_file(latest_ckpt,
    #                                       tensor_name='', all_tensors=True, all_tensor_names=True)
