import tensorflow as tf
import huaytools as hy

import logging

# logger = logging.getLogger(__name__)
# hy.set_logging_basic_config()
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    """DNN with two hidden layers"""
    # input layer
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # hidden layers
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # output layer
    logits = tf.layers.dense(net, params['n_classes'], activation=None)  # linear

    # Compute predictions.
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # change shape from [None] to [None, 1]
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        # predictions = predicted_classes
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy, update_op = tf.metrics.accuracy(labels=labels,
                                              predictions=predicted_classes,
                                              name='acc_op')
    metrics = {'accuracy': (accuracy, update_op)}
    tf.summary.scalar('accuracy', update_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics,
            evaluation_hooks=[tf.train.LoggingTensorHook({'step': tf.train.get_global_step(), 'loss': loss,
                                                          'accuracy': accuracy},
                                                         every_n_iter=1)])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    # train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    # self._init_train_op()
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def input_fn(csv_path, n_epoch=10, n_batch=30, features_only=False):
    """for train and evaluate"""
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    def _parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, record_defaults=[[0.0], [0.0], [0.0], [0.0], [0]])

        # Pack the result into a dictionary
        features = dict(zip(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'],
                            fields))

        # Separate the label from the features
        label = features.pop('Species')

        if features_only:
            return features
        else:
            return features, label

    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat(n_epoch).batch(n_batch)

    # return dataset.make_one_shot_iterator().get_next()
    return dataset


if __name__ == '__main__':
    """"""
    fcs = [tf.feature_column.numeric_column('SepalLength'),
           tf.feature_column.numeric_column('SepalWidth'),
           tf.feature_column.numeric_column('PetalLength'),
           tf.feature_column.numeric_column('PetalWidth')]

    est = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./log',
        params={'feature_columns': fcs,
                'hidden_units': [10, 10],
                'n_classes': 3, })

    est.train(input_fn=lambda: input_fn(csv_path=r"../data/iris/iris_training.csv", n_epoch=100))

    est.evaluate(input_fn=lambda: input_fn(csv_path=r"../data/iris/iris_test.csv", n_epoch=1))

    ret = est.predict(
        input_fn=lambda: input_fn(csv_path=r"../data/iris/iris_test.csv", n_epoch=2, n_batch=10, features_only=True))

    for r in ret:
        print(r)
    # print(est.get_variable_names())
