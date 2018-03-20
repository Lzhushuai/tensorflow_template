import tensorflow as tf

from huaytools.tf_.base import BaseModel


class BaseTrainer(object):
    def __init__(self, sess, model: BaseModel,
                 data, config, summary):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.summary = summary

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        cur_epoch = self.model.cur_epoch.eval(self.sess)

        for cur_epoch in range(cur_epoch, self.config.n_epoch):
            self.train_epoch()
            # self.sess.run(self.model.increment_epoch)
            self.increment_epoch()

    def increment_epoch(self):
        self.sess.run(tf.assign_add(self.model.cur_epoch, 1))

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def train_input_fn(self, *args, **kwargs):
        """
        ref: https://www.tensorflow.org/versions/master/get_started/custom_estimators#write_an_input_function
        """
        raise NotImplementedError