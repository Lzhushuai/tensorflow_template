"""
show how to use `tf.feature_columns`
"""
import os
import logging
import tensorflow as tf
import tensorlayer as tl
import huaytools as hy

from base.base_config import Config
from base.base_model import BaseModel

logger = logging.getLogger(__name__)


class ExampleModelFc(BaseModel):
    """
    Attributes:
        feature_columns(list of tf.feature_column): a new feature of tensorflow.
            If you are not familiar with it, let it be None.
            ref: https://www.tensorflow.org/versions/master/get_started/feature_columns
    """

    def __init__(self, config, feature_columns=None, graph=None):
        super(ExampleModelFc, self).__init__(config, graph)

        self.feature_columns = feature_columns

    def _init_graph(self):
        pass

    def train(self, sess, dataset, *args, **kwargs):
        pass

    def evaluate(self, sess, dataset, *args, **kwargs):
        pass

    def predict(self, sess, dataset, *args, **kwargs):
        pass