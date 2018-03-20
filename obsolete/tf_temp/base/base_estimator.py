import logging
import tensorflow as tf

from huaytools.tf_temp.base.base_model import BaseModel
from huaytools.tf_temp.utils.Config import Config

logger = logging.getLogger(__name__)


class BaseEstimator(object):
    """
    Attributes:
        config(Config): a bunch-type config object
    """

    def __init__(self, config):
        self.config = config

        self.estimator = None
        self._init_estimator()

        self.feature_columns = None
        self._init_feature_columns()

    def model_fn(self, features, labels, mode, params, **kwargs):
        """"""
        raise NotImplementedError

    def input_fn(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def train(self, hooks=None, steps=None, max_steps=None, saving_listeners=None, **kwargs):
        """"""
        return self.estimator.train(input_fn=lambda: self.input_fn(**kwargs),
                                    hooks=hooks, steps=steps, max_steps=max_steps,
                                    saving_listeners=saving_listeners)

    def evaluate(self, steps=None, hooks=None, checkpoint_path=None, name=None, **kwargs):
        """"""
        return self.estimator.evaluate(input_fn=lambda: self.input_fn(**kwargs),
                                       steps=steps, hooks=hooks, checkpoint_path=checkpoint_path, name=name)

    def predict(self, predict_keys=None, hooks=None, checkpoint_path=None, **kwargs):
        """"""
        return self.estimator.predict(input_fn=lambda: self.input_fn(**kwargs),
                                      predict_keys=predict_keys, hooks=hooks, checkpoint_path=checkpoint_path)

    def _init_estimator(self, *args, **kwargs):
        """
        Examples:
            self.estimator = tf.estimator.Estimator(
                model_fn=self.model_fn,
                params={
                    'feature_columns': self.feature_columns,
                    'hidden_units': [10, 10],
                    'n_classes': 3,
                })
        """
        raise NotImplementedError

    def _init_feature_columns(self, *args, **kwargs):
        """"""
        raise NotImplementedError


if __name__ == '__main__':
    """"""


    def t(a, b):
        print(a, b)


    def tt(**kwargs):
        t(**kwargs)


    tt(a=1, b=2)
