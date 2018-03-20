from huaytools.tf_ import BaseConfig
from huaytools.tf_.base import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config: BaseConfig):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def init_input(self):
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.n_feature)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).repeat()
        self.iterator = self.dataset.make_initializable_iterator()
        # sess.run(iterator, feed_dict)

    def build_model(self):
        self.init_input()

        self.is_training = tf.placeholder(tf.bool)

        # self.x = tf.placeholder(tf.float32, shape=[None] + self.config.n_feature)
        # self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="densee2")
        d2 = tf.layers.dense(d1, 10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self, max_to_keep=5,  *args, **kwargs):
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
