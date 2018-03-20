"""
fast get data from ../data/iris
"""
import tensorflow as tf


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])

    # Pack the result into a dictionary
    features = dict(zip(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'],
                        fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def _parse_csv(csv_path, n_epoch=1, n_batch=12, buffer_size=1000):
    dataset = tf.data.TextLineDataset(csv_path).skip(1).map(_parse_line)

    # dataset = dataset.shuffle(buffer_size).repeat(n_epoch).batch(n_batch)
    dataset = dataset.repeat(n_epoch).batch(n_batch)

    # ds_iter = dataset.make_one_shot_iterator()

    return dataset


def get_train_dataset(
        csv_path=r"..\data\iris\iris_training.csv"):
    return _parse_csv(csv_path)


def get_eval_dataset(
        csv_path=r"..\data\iris\iris_test.csv"):
    return _parse_csv(csv_path)


def get_pred_dataset():
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    expected_y = ['Setosa', 'Versicolor', 'Virginica']

    return tf.data.Dataset.from_tensor_slices(predict_x).batch(16).make_one_shot_iterator()


if __name__ == '__main__':
    dataset = get_train_dataset()

    fcs = [
        tf.feature_column.numeric_column('SepalLength'),
        tf.feature_column.numeric_column('SepalWidth'),
        tf.feature_column.numeric_column('PetalLength'),
        tf.feature_column.numeric_column('PetalWidth')
    ]

    ds_iter = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        # print(get_train_ds_iter().output_shapes)
        # print(get_eval_ds_iter().output_shapes)
        # print(get_pred_ds_iter().output_shapes)

        f, l = ds_iter.get_next()

        inputs = tf.feature_column.input_layer(f, fcs)

        print(inputs.eval())

    # features = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    # labels = [2, 3, 4]
    #
    # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # ds_iter = dataset.batch(12).make_one_shot_iterator()
    #
    # with tf.Session() as sess:
    #     print(sess.run(ds_iter.get_next()))
    #
    #     # print(sess.run(tf.decode_csv('1,2,3,4,5', record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])))
