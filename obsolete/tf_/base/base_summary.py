import os

import tensorflow as tf


class BaseSummary(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        self.summary_placeholders = dict()
        self.summary_ops = dict()

        self.summary_writer = None

    def summarize(self, step, summary_dict, scope=None, summarizer="train"):
        summary_dir = os.path.join(self.config.summary_dir, summarizer)
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        with tf.variable_scope(scope or "summarize"):

            summary_list = []

            for tag, value in summary_dict.items():
                if tag not in self.summary_ops:
                    if len(value.shape) <= 1:
                        self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                    else:
                        self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]),
                                                                        name=tag)
                        self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                summary_ret = self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value})
                summary_list.append(summary_ret)

            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)

            self.summary_writer.flush()