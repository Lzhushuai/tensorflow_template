import logging
import tensorflow as tf

from huaytools.tf_temp.utils.Config import Config

logger = logging.getLogger(__name__)


class BaseModel(object):
    """
    Attributes:
        config(Config):
        loss:
        train_op:
        global_step:
        saver(tf.train.Saver):
    """

    def __init__(self, config, graph=None):
        self.config = config
        self.graph = graph

        self._init_global_step()
        self._init_saver()

    def build_graph(self):
        """
        `build_graph` must be overridden. If not, it just run
        the function of base class, not the subclass's.

        Examples:
            ```
            if self.graph is None:
                self.graph = tf.get_default_graph()

            with self.graph.as_default():
                self._init_model()
                self._init_loss()
                self._init_train_op()
            ```
        """
        raise NotImplementedError

    def set_ds_iter(self, ds_iter):
        self.ds_iter = ds_iter

    def _init_input(self):
        """"""
        raise NotImplementedError

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, sess, ds_iter, *args, **kwargs):
        """"""
        raise NotImplementedError

    def evaluate(self, sess, ds_iter, *args, **kwargs):
        """"""
        raise NotImplementedError

    def predict(self, sess, ds_iter, *args, **kwargs):
        """"""
        raise NotImplementedError

    def _init_loss(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def _init_train_op(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def _init_global_step(self, scope=None, **kwargs):
        """"""
        with tf.variable_scope(scope or "global_step"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step', **kwargs)

    def _init_saver(self, *args, **kwargs):
        """"""
        self.saver = tf.train.Saver(*args, **kwargs)

    def load(self, sess, ckpt_dir=None):
        if ckpt_dir is None:
            ckpt_dir = self.config.ckpt_dir
            assert ckpt_dir is not None, "`ckpt_dir` is None!"

        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            logger.info("Loading the latest model from checkpoint: {} ...\n".format(latest_ckpt))
            self.saver.restore(sess, latest_ckpt)
            logger.info("model loaded")
        else:
            logger.info("No model checkpoint\n")

    def save(self, sess, ckpt_dir=None, **kwargs):
        """"""
        if ckpt_dir is None:
            ckpt_dir = self.config.ckpt_dir
            assert ckpt_dir is not None, "`ckpt_dir` is None!"

        logger.info("Saving model...")
        self.saver.save(sess, ckpt_dir, **kwargs)
        logger.info("Model is saved.")


if __name__ == '__main__':
    """"""
    c = Config("aa")

    print(c.aaa)