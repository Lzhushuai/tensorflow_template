import tensorflow as tf

from tensorflow_template.base.base_model import BasicModel, BaseConfig

tf.logging.set_verbosity(tf.logging.INFO)


class MnistModel(BasicModel):
    def _init_model(self, features, labels, mode):
        # For tf newer, the most important is to know the in/out shape of each layer
        # Input size:  [batch_size, 28, 28, 1]
        # Output size: [batch_size, 28, 28, 32]
        # The strides is (1, 1) default, so it will not change
        # the size of image which is (28, 28) when `padding="same"`
        conv1 = tf.layers.conv2d(features,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # In:  [batch_size, 28, 28, 32]
        # Out: [batch_size, 14, 14, 32], here 14 = 28/strides = 28/2 = 14
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        # In:  [batch_size, 14, 14, 32]
        # Out: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # In:  [batch_size, 14, 14, 64]
        # Out: [batch_size, 7, 7, 64], where 7 is computed same as pool1
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        # In:  [batch_size, 7, 7, 64]
        # Out: [batch_size, 7*7*64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # In:  [batch_size, 7*7*64]
        # Out: [batch_size, 1024]
        dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)

        # Use dropout only when (mode == TRAIN)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == self.ModeKeys.TRAIN))

        # The out of model which has not been softmax
        self._logits = tf.layers.dense(inputs=dropout, units=10)
        self._predictions = tf.argmax(self._logits, axis=1)

        if mode == self.ModeKeys.PREDICT:
            # return self._predictions
            return {'predictions': self._predictions,
                    'probability': tf.nn.softmax(self._logits)}

        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)
        _, self._accuracy = tf.metrics.accuracy(labels, self._predictions)

        self.add_log_tensors({'loss': self._loss,
                              'acc': self._accuracy})

        if mode == self.ModeKeys.EVALUATE:
            return [self._loss, self._accuracy]

        self._train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
            self._loss, global_step=tf.train.get_global_step())

        if mode == self.ModeKeys.TRAIN:
            return [self._train_op, self._loss, self._accuracy]


def get_config():
    config = BaseConfig('mnist')

    return config


from examples.mnist.data_helper import *

if __name__ == '__main__':
    """"""

    config = get_config()

    ds_train, ds_test, ds_pred = get_dataset()

    config.n_epoch_train = 1
    config.n_epoch_eval = 1
    config.max_steps = 250

    model = MnistModel(config)

    n_epoch = 100

    if model.get_global_step() < 250:
        for i in range(1, n_epoch+1):

            model.train(ds_train)

            if i % 10 == 0:
                model.evaluate(ds_test)
    else:
        model.evaluate(ds_test)

    ret = model.predict(ds_pred)

    for i in ret:
        print(i)
