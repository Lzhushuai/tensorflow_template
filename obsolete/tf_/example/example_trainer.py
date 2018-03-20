from huaytools.tf_.base.base_trainer import BaseTrainer
from huaytools.tf_ import BaseModel
from tqdm import tqdm
import numpy as np

import tensorflow as tf


class ExampleTrainer(BaseTrainer):
    def __init__(self, sess, model: BaseModel,
                 data, config, summary):
        super(ExampleTrainer, self).__init__(sess, model, data, config, summary)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.n_batch))

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def train_epoch(self):
        loop = tqdm(range(self.config.n_step))
        losses = []
        accs = []
        for it in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step.eval(self.sess)

        summaries_dict = dict()
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc

        self.summary.summarize(cur_it, summary_dict=summaries_dict)
        self.model.save(self.sess, self.config.ckpt_dir)

    def train_input_fn(self, features, labels, buffer_size=1000):
        """
        ref: https://www.tensorflow.org/versions/master/get_started/custom_estimators#write_an_input_function
        Args:
            features:
            labels:

        Returns:

        """
        # Convert the inputs to a Dataset.
        # PS: the example features is a dictionary features
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(buffer_size).repeat().batch(self.config.n_batch)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_input()
