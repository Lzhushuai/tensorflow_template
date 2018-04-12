import os
import math

import tensorflow as tf
from tensorflow_template.base.base_model import BaseModel, BaseConfig

tf.logging.set_verbosity(tf.logging.INFO)


class CBOW(BaseModel):
    """"""

    def _init_model(self, features, labels, mode):
        """"""
        # init the embedding
        embeddings = tf.Variable(
            tf.random_uniform([self.config.vocabulary_size, self.config.embedding_size], minval=-1.0, maxval=1.0))

        with tf.name_scope("sampled_softmax_loss"):
            W = tf.Variable(tf.truncated_normal([self.config.vocabulary_size, self.config.embedding_size],
                                                stddev=1.0 / math.sqrt(self.config.embedding_size)))
            b = tf.Variable(tf.zeros([self.config.vocabulary_size]))

        context_embed = []
        for i in range(2 * self.config.skip_window):
            context_embed.append(tf.nn.embedding_lookup(embeddings, features[:, i]))
        avg_context_embed = tf.reduce_mean(tf.stack(context_embed), axis=0, keep_dims=False)

        self._loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(W, b,
                                       labels=labels,
                                       inputs=avg_context_embed,
                                       num_sampled=self.config.num_sampled,
                                       num_classes=self.config.vocabulary_size))

        self._train_op = tf.train.AdadeltaOptimizer(self.config.learning_rate).minimize(self._loss,
                                                                                        global_step=tf.train.get_global_step())

        # normalize the embeddings
        self.embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        # 中间测试
        self.valid = tf.constant(config.valid, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(self.embeddings, self.valid)
        self.similarity = tf.matmul(valid_embeddings, tf.transpose(self.embeddings))

    def train(self, dataset):
        """"""
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            average_loss = 0
            for step in range(config.n_steps):
                features, labels = next(dataset)
                self._init_model(features, labels, mode=self.ModeKeys.TRAIN)

                _, loss_val = sess.run([self._train_op, self._loss])

                average_loss += loss_val

                if (step + 1) % 1000 == 0:
                    average_loss = average_loss / step
                    tf.logging.info('Average loss at step %d: %f', step + 1, average_loss)

                # 中间测试
                if (step + 1) % 10000 == 0:
                    sim = model.similarity.eval()
                    for i in range(config.valid_size):
                        valid_word = id2word[valid_data[i]]
                        top_k = 10
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = id2word[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        tf.logging.info(log)

    def get_embedding(self):
        """"""


# def get_config():
#     config = BaseConfig()
#
#     config.n_batch_train = 64  # batch size for train
#     config.embedding_size = 128
#     config.vocabulary_size = 50000
#     config.skip_window = 1
#     config.num_skips = 2
#     config.num_sampled = 64  # 负采样
#
#     return config


class get_config(BaseConfig):
    def __init__(self):
        super(get_config, self).__init__()

        self.n_batch_train = 64  # batch size for train
        self.embedding_size = 128
        self.vocabulary_size = 50000
        self.skip_window = 1
        # self.num_skips = 2
        self.num_sampled = 64  # negative sample number


if __name__ == '__main__':
    """"""
    config = get_config()

    from examples.nlp.word2vec.data_helper import get_all_data

    filename = r"D:\OneDrive\workspace\data\nlp\text8\text8"

    batch_iter, valid_data, word2id, id2word = get_all_data(filename,
                                                            config.n_batch_train, config.skip_window)

    config.valid_data = valid_data

    model = CBOW(config)

    model.train(ds_train)
