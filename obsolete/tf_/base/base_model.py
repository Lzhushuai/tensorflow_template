import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseModel(object):
    def __init__(self, config):
        """"""
        self.config = config

        self.saver = None
        self.init_saver()

        self.global_step = None
        self.init_global_step()

        self.cur_epoch = None
        self.init_cur_epoch()

    def build_model(self):
        raise NotImplementedError

    def init_saver(self, max_to_keep=5, *args, **kwargs):
        """
        Generally, all need for init_saver is
            `self.saver = tf.train.Saver(max_to_keep=max_to_keep)`

            For `tf.train.Saver` has many other parameter, such as `max_to_keep`,
            you can override the `init_saver()` to use them.
        """
        raise NotImplementedError

    def init_global_step(self):
        with tf.variable_scope("global_step"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def init_cur_epoch(self):
        with tf.variable_scope("cur_epoch"):
            self.cur_epoch = tf.Variable(0, trainable=False, name='cur_epoch')

    def load(self, sess, ckpt_dir):
        """load checkpoint from `ckpt_dir`"""
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            logger.info("Loading the latest model checkpoint {} ...\n".format(latest_ckpt))
            self.saver.restore(sess, latest_ckpt)
            logger.info("model loaded")
        else:
            logger.info("No model checkpoint\n")

    def save(self, sess, ckpt_dir):
        """save checkpoint to `ckpt_dir`"""
        logger.info("Saving model...")
        self.saver.save(sess, ckpt_dir, self.global_step)
        logger.info("Model saved")


if __name__ == '__main__':
    """"""

    tf.estimator.LinearClassifier