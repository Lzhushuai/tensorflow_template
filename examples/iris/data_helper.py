import csv
import tensorflow as tf
import numpy as np


def read_csv(csv_path, features_only=False):
    with open(csv_path, newline='') as f:
        next(f)  # skip the first line
        reader = csv.reader(f)
        features, labels = [], []
        for line in reader:
            feature, label = [float(i) for i in line[:-1]], int(line[-1])
            # yield feature, label
            features.append(feature)
            labels.append(label)

        if features_only:
            return features
        else:
            return features, labels


def get_dataset(mode, features_only=False):
    if mode == 'train':
        # return tf.data.Dataset.from_generator(
        #     lambda: csv_yield(r"../data/iris/iris_training.csv"),
        #     output_types=(tf.float32, tf.int32))
        return tf.data.Dataset.from_tensor_slices(read_csv(r"../data/iris/iris_training.csv", features_only))

    elif mode == 'eval':
        # return tf.data.Dataset.from_generator(
        #     lambda: read_csv(r"../data/iris/iris_test.csv"),
        #     output_types=(tf.float32, tf.int32))
        return tf.data.Dataset.from_tensor_slices(read_csv(r"../data/iris/iris_test.csv", features_only))
    else:
        # data = [[5.1, 5.9, 6.9], [3.3, 3.0, 3.1], [1.7, 4.2, 5.4], [0.5, 1.5, 2.1]]
        # data = np.array(data).T  # transposition
        # expected_y = ['Setosa', 'Versicolor', 'Virginica']  # []

        data = [[6.4, 2.8, 5.6, 2.2], [5., 2.3, 3.3, 1.], [4.9, 2.5, 4.5, 1.7]]
        expected_y = [2, 1, 2]
        return tf.data.Dataset.from_tensor_slices(data), expected_y


if __name__ == '__main__':
    # ds = tf.data.Dataset.from_generator(
    #     lambda: csv_yield(r"../data/iris/iris_training.csv"),
    #     output_types=(tf.float32, tf.int32)).batch(12)

    ds = tf.data.Dataset.from_tensor_slices(read_csv(r"../data/iris/iris_training.csv"))
    print(ds.output_shapes)

    ds_iter = ds.make_one_shot_iterator()
    print(ds_iter.output_shapes)

    # each column is a sample
    data = [[5.1, 5.9, 6.9],
            [3.3, 3.0, 3.1],
            [1.7, 4.2, 5.4],
            [0.5, 1.5, 2.1]]
    data = np.array(data).T
    ds2_iter = tf.data.Dataset.from_tensor_slices(data).batch(12).make_one_shot_iterator()

    with tf.Session() as sess:
        print(sess.run(ds_iter.get_next()))
        print(sess.run(ds2_iter.get_next()))
