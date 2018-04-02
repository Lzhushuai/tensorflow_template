import os

import tensorflow as tf
from tensorflow_template.base.base_model import BaseModel, BaseConfig


class CnnSentenceModel(BaseModel):

    # def _init_model(self, features, labels, mode):
    #     """"""
    #     with tf.name_scope('embedding'):
    #         # Here use the random weights and adjust it when training.
    #         # If the original dataset is small or you want to apply the model to a wide application,
    #         # it should use some pre-trained word embedding, such as word2vec, GloVe or FastText.
    #         W = tf.Variable(
    #             tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
    #             name="W")
    #         embedded_chars = tf.nn.embedding_lookup(W, features)
    #         embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    #         # shape (x, y) -> (x, y, 1)
    #         # The last dim expanded represent `channel` in the CNN input
    #
    #     # Create cnn for different filter size:
    #     pooled_outputs = []
    #     for i, filter_size in enumerate(self.config.filter_sizes):
    #         with tf.name_scope("cnn-%i" % filter_size):
    #             filter_shape = [filter_size, self.config.embedding_size, 1, self.config.n_filter]
    #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    #             b = tf.Variable(tf.constant(0.1, shape=[self.config.n_filter]), name="b")
    #             conv = tf.nn.conv2d(embedded_chars_expanded,
    #                                 W,
    #                                 strides=[1, 1, 1, 1],
    #                                 padding="VALID",
    #                                 name="conv")
    #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    #             pooled = tf.nn.max_pool(h,
    #                                     ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
    #                                     strides=[1, 1, 1, 1],
    #                                     padding='VALID',
    #                                     name="pool")
    #             pooled_outputs.append(pooled)
    #
    #     num_filters_total = self.config.n_filter * len(self.config.filter_sizes)
    #     h_pool = tf.concat(pooled_outputs, 3)
    #     h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    #
    #     # Add dropout
    #     with tf.name_scope("dropout"):
    #         h_drop = tf.nn.dropout(h_pool_flat, self.config.dropout_keep_prob)
    #
    #     # Final (un-normalized) scores and predictions
    #     l2_loss = tf.constant(0.0)
    #     with tf.name_scope("output"):
    #         W = tf.get_variable("W",
    #                             shape=[num_filters_total, self.config.n_class],
    #                             initializer=tf.contrib.layers.xavier_initializer())
    #         b = tf.Variable(tf.constant(0.1, shape=[self.config.n_class]), name="b")
    #         l2_loss += tf.nn.l2_loss(W)
    #         l2_loss += tf.nn.l2_loss(b)
    #         self._logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    #         self._predictions = tf.argmax(self._logits, 1, name="predictions")
    #
    #     if mode == self.ModeKeys.PREDICT:
    #         return {'predictions': self._predictions,
    #                 'scores': tf.nn.softmax(self._logits)}
    #
    #     # Calculate mean cross-entropy loss
    #     with tf.name_scope("loss"):
    #         losses = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=labels)
    #         self._loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss
    #
    #     # Accuracy
    #     with tf.name_scope("accuracy"):
    #         correct_predictions = tf.equal(self._predictions, tf.argmax(labels, 1))
    #         self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    #
    #     self.add_log_tensors({'loss': self._loss, 'acc': self._accuracy})
    #     if mode == self.ModeKeys.EVALUATE:
    #         return [self._loss, self._accuracy]
    #
    #     # Train_op
    #     with tf.name_scope('train_op'):
    #         optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
    #         grads_and_vars = optimizer.compute_gradients(self._loss)
    #         self._train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
    #
    #     return [self._train_op, self._loss, self._accuracy]

    def _embedding(self, inputs):
        with tf.name_scope('embedding'):
            # Here use the random weights and adjust it when training.
            # If the original dataset is small or you want to apply the model to a wide application,
            # it should use some pre-trained word embedding, such as word2vec, GloVe or FastText.
            self._W = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
                name="W")
            self._embedded_chars = tf.nn.embedding_lookup(self._W, inputs)
            self._embedded_chars_expanded = tf.expand_dims(self._embedded_chars, -1)

    def _cnn(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("cnn-%i" % filter_size):
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.n_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.n_filter]), name="b")
                conv = tf.nn.conv2d(self._embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)

        self._num_filters_total = self.config.n_filter * len(self.config.filter_sizes)
        self._h_pool = tf.concat(pooled_outputs, 3)
        self._h_pool_flat = tf.reshape(self._h_pool, [-1, self._num_filters_total])

    def _dropout(self):
        with tf.name_scope("dropout"):
            self._h_drop = tf.nn.dropout(self._h_pool_flat, self.config.dropout_keep_prob)

    def _output(self):
        self._l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[self._num_filters_total, self.config.n_class],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.n_class]), name="b")
            self._l2_loss += tf.nn.l2_loss(W)
            self._l2_loss += tf.nn.l2_loss(b)
            self._logits = tf.nn.xw_plus_b(self._h_drop, W, b, name="scores")
            self._predictions = tf.argmax(self._logits, 1, name="predictions")

    def _loss_op(self, labels):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=labels)
            self._loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self._l2_loss

    def _acc_op(self, labels):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self._predictions, tf.argmax(labels, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _init_model(self, features, labels, mode):
        self._embedding(features)

        self._cnn()

        self._dropout()

        self._output()
        if mode == self.ModeKeys.PREDICT:
            # return [self._predictions]
            return {'predictions': self._predictions,
                    'scores': tf.nn.softmax(self._logits)}

        self._loss_op(labels)
        self._acc_op(labels)

        # self.add_log_tensor('loss', self._loss)
        # self.add_log_tensor('acc', self._accuracy)
        self.add_log_tensors({'loss': self._loss,
                              'acc': self._accuracy})
        if mode == self.ModeKeys.EVALUATE:
            return [self._loss, self._accuracy]

        with tf.name_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self._loss)
            self._train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return [self._train_op, self._loss, self._accuracy]


def get_config():
    config = BaseConfig('cnn_sent')
    config.n_class = 2
    config.shuffle_buffer_size = 5000
    config.log_n_steps_train = 1
    config.n_batch_train = 256
    config.n_epoch_train = 20
    config.learning_rate = 1e-3

    # cnn
    config.vocab_size = 50000
    config.sequence_length = 56  # x_train.shape[1]
    config.embedding_size = 300
    config.filter_sizes = [3, 4, 5]
    config.n_filter = 128

    config.dropout_keep_prob = 0.5
    config.l2_reg_lambda = 0.0

    return config


def get_ds():
    from examples.nlp.cnn_sentence_classification.data_helper import get_dataset

    x_train, x_test, y_train, y_test, vocab_processor = get_dataset()
    vocab_processor.save(os.path.join(config.out_dir, "vocab.txt"))

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_pred = tf.data.Dataset.from_tensor_slices(x_test)

    tf.logging.info("output shape of ds_train is {}".format(ds_train.output_shapes))
    tf.logging.info("output shape of ds_test is {}".format(ds_test.output_shapes))
    tf.logging.info("output shape of ds_pred is {}".format(ds_pred.output_shapes))

    return ds_train, ds_test, ds_pred


if __name__ == '__main__':
    """"""
    config = get_config()

    ds_train, ds_test, ds_pred = get_ds()

    model = CnnSentenceModel(config)

    model.train(ds_train)

    model.evaluate(ds_test)

    for i in model.predict(ds_pred):
        print(i)
