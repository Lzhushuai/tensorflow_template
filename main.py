import numpy as np
import tensorflow as tf


class ExModel(object):
    def __init__(self, graph=None):

        if graph is None:
            self.graph = tf.get_default_graph()
        else:
            self.graph = graph

        self.sess = tf.Session(graph=self.graph)

        self.build_model()

        print(self.sess.run(tf.report_uninitialized_variables()))  # [b'global_step']
        print(self.sess.run(self._global_step))  # uninitialized error

    def build_model(self):
        with self.graph.as_default():
            self._global_step = tf.Variable(0, trainable=False, name='global_step')

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)

if __name__ == '__main__':
    model = ExModel()

    graph = tf.get_default_graph()

    sess = tf.Session(graph=graph)

    with graph.as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # sess.run(tf.global_variables_initializer())
    print(sess.run(global_step))

