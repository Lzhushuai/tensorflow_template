import logging
import tensorflow as tf

from base.base_config import Config
# from base.base_mode import ModeKeys

logger = logging.getLogger(__name__)


class BaseModel(object):
    """
    Attributes:
        config(Config): The bunch config of model.
        graph(tf.Graph): The graph of model, default to use the `tf.get_default_graph()`.
        mode(ModeKeys): Default is `ModeKeys.TRAIN`. Other mode are `EVAL`, `PREDICT` and `RETRAIN`
            `RETRAIN` means load from a checkpoint, then it need not to run the `self.init_op` again.
            Actually, this attr has no use at this version.
    """
    class ModeKeys(object):
        """
        Standard names for model modes.

            * `TRAIN`: training mode.
            * `EVAL`: evaluation mode.
            * `PREDICT`: inference mode.
            * `RETRAIN`: retrain mode(new added).

            ref: https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/ModeKeys
        """

        TRAIN = 'train'
        EVAL = 'eval/test/validate'
        PREDICT = 'infer/predict'
        RETRAIN = 'retrain'

    def __init__(self, config, graph=None):
        self.config = config

        if graph is None:
            self.graph = tf.get_default_graph()
        else:
            self.graph = graph

        self.sess = tf.Session(graph=self.graph, config=self.config.sess_config)

        self.mode = self.ModeKeys.TRAIN

        self.build_model()
        self.init_global_variables()

    def build_model(self):
        """All the variable you want to save should be under the `self.graph`."""
        with self.graph.as_default():
            self._init_global_step()
            self._init_graph()
            self._init_saver()

    def init_global_variables(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        logger.debug("Uninitialized_variables are {}".format(self.sess.run(tf.report_uninitialized_variables())))

    def _init_graph(self):
        """
        The basic part:
            0. the `tf.placeholder`
            1. the net
            2. the output (self.logits & self.prediction)
            3. the loss (self.loss)
            4. the train_op (self.train_op)
        other:
            the metrics(such as `tf.metrics.accuracy`)
            the summary(ref `tf.summary.FileWriter`)

        Examples:
            ```
            self.features = tf.placeholder(tf.float32, [None] + self.config.n_feature, 'features')
            self.labels = tf.placeholder(tf.int32, [None], 'labels')

            net = self.features  # input_layer
            for units in self.config.n_units:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
                # net = tf.layers.Dense(units=self.config.n_units[0], activation=tf.nn.relu)(net)

            self.logits = tf.layers.dense(net, self.config.n_class, activation=None)
            self.prediction = tf.argmax(self.logits, axis=1)

            self.accuracy, self.update_op = tf.metrics.accuracy(labels=self.labels,
                                                                predictions=self.prediction,
                                                                name='acc_op')

            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self._global_step)
            ```
        """

        raise NotImplementedError

    def train(self, dataset, *args, **kwargs):
        """
        Examples:
            ```
            if self.mode == self.ModeKeys.TRAIN:
                sess.run(self.init_op)

            for _ in range(self.config.n_epoch):
                ds_iter = dataset.shuffle(buffer_size).batch(self.config.n_batch).make_one_shot_iterator()
                while True:
                    try:
                        features, labels = sess.run(ds_iter.get_next())
                        loss_val, _ = sess.run([self.loss, self.train_op], feed_dict={self.features: features,
                                                                                      self.labels: labels})
                        logger.info("The loss of Step {} is {}".format(self.global_step, loss_val))
                    except tf.errors.OutOfRangeError:
                        break
                self.save(sess)
            ```

        Args:
            dataset(tf.data.Dataset): need to yield a tuple (features, labels)
                `features, labels = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:

        """
        raise NotImplementedError

    def evaluate(self, dataset, *args, **kwargs):
        """

        Args:
            dataset(tf.data.Dataset): need to yield a tuple (features, labels) (same as train)
                `features, labels = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:
            the metrics of
        """
        raise NotImplementedError

    def predict(self, dataset, *args, **kwargs):
        """

        Args:
            dataset(tf.data.Dataset): need to yield only the features
                `features = ds_iter.get_next()`
            *args: reserve
            **kwargs: reserve

        Returns:
            the result of predict

        """
        raise NotImplementedError

    def _init_global_step(self, name='global_step', **kwargs):
        """"""
        self._global_step = tf.Variable(0, trainable=False, name=name, **kwargs)

    def _init_saver(self, *args, **kwargs):
        """
        The saver must be define under the `self.graph` and init at the last of the graph,
            otherwise it can't find the variables under the graph.
        Just copy the the examples to the subclass

        Examples:
            ```
            self.saver = tf.train.Saver(*args, **kwargs)
            ```
        """
        self.saver = tf.train.Saver(*args, **kwargs)

    def load(self, ckpt_dir=None):
        """"""
        if ckpt_dir is None:
            ckpt_dir = self.config.ckpt_dir
            assert ckpt_dir is not None, "`ckpt_dir` is None!"

        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            # sess.run(self._init_op)
            logger.info("Loading the latest model from checkpoint: {} ...\n".format(latest_ckpt))
            self.saver.restore(self.sess, latest_ckpt)
            logger.info("model loaded")
            self.mode = self.ModeKeys.RETRAIN
        else:
            logger.info("No model checkpoint\n")

    def save(self, ckpt_dir=None, **kwargs):
        """"""
        if ckpt_dir is None:
            ckpt_dir = self.config.ckpt_dir
            assert ckpt_dir is not None, "`ckpt_dir` is None!"

        logger.info("Saving model...")
        self.saver.save(self.sess, ckpt_dir, self._global_step, **kwargs)
        logger.info("Model is saved.")

    @property
    def global_step(self):
        return self._global_step.eval(self.sess)


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








































