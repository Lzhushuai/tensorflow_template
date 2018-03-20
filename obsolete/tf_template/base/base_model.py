import logging
import tensorflow as tf

from huaytools.tf_temp.utils.Config import Config

logger = logging.getLogger(__name__)


class BaseModel(object):
    """
    Attributes:
        config(Config): The bunch config of model.
        graph(tf.Graph): The graph of model, default to use the `tf.get_default_graph()`.
        feature_columns(list of tf.feature_column): a new feature of tensorflow.
            If you are not familiar with it, let it be None.
            ref: https://www.tensorflow.org/versions/master/get_started/feature_columns
    """

    def __init__(self, config, feature_columns=None, graph=None):
        self.config = config
        self.feature_columns = feature_columns

        if graph is None:
            self.graph = tf.get_default_graph()

        self.ds_iter = None
        self._init_global_step()
        self.saver = None
        # self._init_saver()

    def set_ds_iter(self, ds_iter):
        self.ds_iter = ds_iter

    def _init_graph(self, features, labels=None):
        """
        The basic part:
            1. the model/net
            2. the output(self.logits & self.prediction)
            3. the loss(self.loss)
            4. the train_op(self.train_op)
            5. **init the saver**
        other:
            the metrics(such as `tf.metrics.accuracy`)
            the summary(ref `tf.summary.FileWriter`)

        Examples:
            ```
            with self.graph.as_default():
                net = tf.feature_column.input_layer(features, self.feature_columns)

                for units in self.config.n_units:
                    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

                self.logits = tf.layers.dense(net, self.config.n_class, activation=None)

                self.prediction = tf.argmax(self.logits, axis=1)

                if labels is not None:  # train or evaluate
                    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)

                    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
                    self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            ```
        """

        raise NotImplementedError

    def train(self, sess, ds_iter, *args, **kwargs):
        """

        Args:
            sess(tf.Session):
            ds_iter(tf.data.Iterator): need to yield both the features and labels
                `features, labels = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:

        """
        raise NotImplementedError

    def evaluate(self, sess, ds_iter, *args, **kwargs):
        """

        Args:
            sess(tf.Session):
            ds_iter(tf.data.Iterator): need to yield both the features and labels (same as train)
                `features, labels = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:
            the metrics of
        """
        raise NotImplementedError

    def predict(self, sess, ds_iter, *args, **kwargs):
        """

        Args:
            sess(tf.Session):
            ds_iter(tf.data.Iterator): need to yield only the features
                `features = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:
            the result of predict

        """
        raise NotImplementedError

    def _init_global_step(self, name='global_step', **kwargs):
        """
        All the variable you want to save should be under the `self.graph`.
        """
        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, name=name, **kwargs)

    def _init_saver(self, *args, **kwargs):
        """
        The saver must be under the `self.graph` and init at the last of the graph,
            otherwise it can't find the variables under the graph.
        Just copy the the examples in the subclass
            If implement it in the base class, it may not be intellisensed in the subclass.
            Unless you change the function name to `init_saver()`, but I think it should be a private function.

        Examples:
            ```
            with self.graph.as_default():
                self.saver = tf.train.Saver(*args, **kwargs)
            ```
        """
        raise NotImplementedError

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
        self.saver.save(sess, ckpt_dir, self.global_step, **kwargs)
        logger.info("Model is saved.")


if __name__ == '__main__':
    class A:
        def fun(self):
            print("A")

        def bar(self):
            self.fun()

    class B(A):
        def fun(self):
            print("B")

    b = B()
    b.bar()








































