import logging
import huaytools as hy
import tensorflow as tf

from tensorflow_template.base.base_model import BasicModel, BaseConfig

# logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
# hy.set_logging_basic_config(level=logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.INFO)


# print(tf.logging.get_verbosity())


class IrisModel(BasicModel):
    def _init_model(self, features, labels, mode):

        # 1. define the net
        # for units in self.config.n_units:
        #     net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.layers.dense(features, units=self.config.n_units[0], activation=tf.nn.relu)
        net = tf.layers.dense(net, units=self.config.n_units[1], activation=tf.nn.relu)

        # the output
        self._logits = tf.layers.dense(net, self.config.n_class, activation=None)
        self._prediction = tf.argmax(self._logits, axis=1)

        if mode == self.ModeKeys.PREDICT:
            self._probabilities = tf.nn.softmax(self._logits)
            # return [self._prediction, self._probabilities]
            # It's better to use dict
            return {'prediction': self._prediction,
                    'probabilities': self._probabilities}

        # 2. define the loss
        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)
        # self.add_log_tensor('loss', self._loss)

        # 3. define the metrics if needed
        _, self._accuracy = tf.metrics.accuracy(labels=labels,
                                                predictions=self._prediction, name='acc_op')
        # self.add_log_tensor('acc', self._accuracy)

        self.add_log_tensors({'loss': self._loss,
                              'acc': self._accuracy})

        if mode == self.ModeKeys.EVALUATE:
            # self.add_metric_ops('acc', self._accuracy)
            return [self._loss, self._accuracy]

        if mode == self.ModeKeys.TRAIN:
            # 4. define the train_op when the mode is TRAIN
            self._optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            self._train_op = self._optimizer.minimize(self._loss, global_step=tf.train.get_global_step())

            return [self._train_op, self._loss, self._accuracy]


def get_config():
    config = BaseConfig('iris')
    config.n_units = [32, 64]
    config.n_class = 3

    # config.n_batch = 20
    # config.eval_n_epoch = 2
    #
    # config.pred_n_epoch = 2
    # config.pred_n_batch = 3

    config.max_steps = 500

    return config


if __name__ == '__main__':
    """"""
    config = get_config()

    model = IrisModel(config)

    from examples.iris.data_helper import get_dataset

    ds_train = get_dataset('train')  # ((4,), ())
    ds_eval = get_dataset('eval')  # ((4,), ())
    # ds_predict, expected_y = get_dataset('predict')  # (4,)
    ds_predict = get_dataset('eval', features_only=True)  # (4,)

    model.train(ds_train)

    model.evaluate(ds_eval)

    ret = model.predict(ds_predict)

    for r in ret:
        print(r)
