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
        self.train_op = self.optimizer.minimize(self.loss, global_step=self._global_step)

    def train(self, dataset, buffer_size=1000, *args, **kwargs):
        # if self.mode == self.ModeKeys.TRAIN:
        #     self.sess.run(self._init_op)  # BUG: it means every it train it will init all the var
        # else:
        #     self.mode = self.ModeKeys.RETRAIN

        for _ in range(self.config.n_epoch):
            ds_iter = dataset.shuffle(buffer_size).batch(self.config.n_batch).make_one_shot_iterator()
            while True:
                try:
                    features, labels = self.sess.run(ds_iter.get_next())
                    loss_val, _, acc_val, _ = self.sess.run([self.loss, self.train_op, self.accuracy, self.update_op],
                                                            feed_dict={self.features: features, self.labels: labels})
                    logger.info("Step {}: loss {}, accuracy {:.3}".format(self.global_step, loss_val, acc_val))
                except tf.errors.OutOfRangeError:
                    break
            self.save()

    def evaluate(self, dataset, *args, **kwargs):
        self.mode = self.ModeKeys.EVAL
        ds_iter = dataset.shuffle(1000).batch(1).make_one_shot_iterator()

        acc_ret = dict()
        i = 1
        while True:
            try:
                features, labels = self.sess.run(ds_iter.get_next())
                prediction, _ = self.sess.run([self.prediction, self.update_op],
                                              feed_dict={self.features: features, self.labels: labels})
                logger.debug("labels is {}, prediction is {}".format(labels, prediction))
                # it's better to run `update_op` first, then run `accuracy`
                accuracy_val = self.sess.run(self.accuracy)
                logger.info('Accuracy is {:.3} of {} test samples'.format(accuracy_val, i))
                acc_ret[i] = accuracy_val
                i += 1
            except tf.errors.OutOfRangeError:
                break

        return acc_ret

    def predict(self, dataset, *args, **kwargs):
        self.mode = self.ModeKeys.PREDICT
        ds_iter = dataset.shuffle(1000).batch(1).make_one_shot_iterator()

        pred_ret = []
        i = 1
        while True:
            try:
                features = self.sess.run(ds_iter.get_next())
                prediction = self.sess.run(self.prediction, feed_dict={self.features: features})
                pred_ret.append(prediction)
                logger.info("the prediction of No.{} is {}".format(i, prediction))
            except tf.errors.OutOfRangeError:
                break

        return np.array(pred_ret).flatten()


if __name__ == '__main__':
    # logger.setLevel(logging.DEBUG)

    config = Config('ex', [4], 3)
    config.ckpt_dir = "D:/Tmp/log/example_ckpt/"
    config.n_epoch = 10
    config.n_feature = [4]
    config.n_units = [10, 10]
    config.n_class = 3

    model = ExampleModel(config)

    from utils.data_iris import *

    ds_train = get_dataset('train')
    ds_eval = get_dataset('eval')
    ds_predict = get_dataset('predict')

    logger.debug(model.global_step)

    model.load()

    logger.debug(model.global_step)

    # model.train(ds_train)
    acc_ret = model.evaluate(ds_eval)
    print(acc_ret)

    pred_ret = model.predict(ds_predict)
    print(pred_ret)


    print('\n\n\n')

    # from tensorflow.python.tools import inspect_checkpoint as chkp
    #
    # latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir)
    #
    # chkp.print_tensors_in_checkpoint_file(latest_ckpt,
    #                                       tensor_name='', all_tensors=True, all_tensor_names=True)
