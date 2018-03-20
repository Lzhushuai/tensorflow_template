import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self):
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]


if __name__ == '__main__':
    data = DataGenerator()
    batch_x, batch_y = next(data.next_batch(2))
    print(batch_x, batch_y)

    ds_iter = tf.data.Dataset.from_tensor_slices([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]).repeat(2).batch(12).make_one_shot_iterator()

    with tf.Session() as sess:
        print(ds_iter.get_next().eval())
