import tensorflow as tf

from huaytools.tf_temp.utils.Config import Config
from huaytools.tf_temp.base.base_estimator import BaseEstimator


class ExampleEstimator(BaseEstimator):
    """"""

    def __init__(self, config):
        super(ExampleEstimator, self).__init__(config)

    def model_fn(self, features, labels, mode, params, **kwargs):
        """DNN with two hidden layers"""
        self.features = features
        self.labels = labels

        # input layer
        net = tf.feature_column.input_layer(features, params['feature_columns'])

        # hidden layers
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        # output layer
        self.logits = tf.layers.dense(net, params['n_classes'], activation=None)  # linear

        # Compute predictions.
        predicted_classes = tf.argmax(self.logits, axis=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],  # change shape from [None] to [None, 1]
                'probabilities': tf.nn.softmax(self.logits),
                'logits': self.logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Compute loss.
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # self._init_loss()
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        # Compute evaluation metrics.
        accuracy, update_op = tf.metrics.accuracy(labels=labels,
                                                  predictions=predicted_classes,
                                                  name='acc_op')
        metrics = {'accuracy': (accuracy, update_op)}
        tf.summary.scalar('accuracy', update_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=self.loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        # train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        # self._init_train_op()
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

    def input_fn(self, csv_path, n_epoch=None, **kwargs):
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

            return features, label

        dataset = dataset.map(_parse_line)

        # Shuffle, repeat, and batch the examples.
        if n_epoch is None:
            n_epoch = self.config.n_epoch
        dataset = dataset.shuffle(1000).repeat(n_epoch).batch(self.config.n_batch)

        # return dataset.make_one_shot_iterator().get_next()
        return dataset

    def pred_input_fn(self, features=None):
        """"""
        if features is None:
            features = self.features
        return tf.data.Dataset.from_tensor_slices(features).batch(16)

    def predict(self, predict_keys=None, hooks=None, checkpoint_path=None, **kwargs):
        return self.estimator.predict(input_fn=lambda: self.pred_input_fn(**kwargs),
                                      predict_keys=predict_keys, hooks=hooks, checkpoint_path=checkpoint_path)

    def _init_estimator(self):
        self._init_feature_columns()

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={
                'feature_columns': self.feature_columns,
                'hidden_units': [10, 10],
                'n_classes': 3,
            })

    def _init_feature_columns(self, *args, **kwargs):
        self.feature_columns = [
            tf.feature_column.numeric_column('SepalLength'),
            tf.feature_column.numeric_column('SepalWidth'),
            tf.feature_column.numeric_column('PetalLength'),
            tf.feature_column.numeric_column('PetalWidth')
        ]


if __name__ == '__main__':
    config = Config("example", n_epoch=300)

    est = ExampleEstimator(config)

    est.train(csv_path=r"D:\OneDrive\workspace\proj\huaytools\huaytools\tf_template\data\iris\iris_training.csv")

    eval_ret = est.evaluate(
        csv_path=r"D:\OneDrive\workspace\proj\huaytools\huaytools\tf_template\data\iris\iris_test.csv",
        n_epoch=1)

    print(eval_ret)

    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    expected_y = ['Setosa', 'Versicolor', 'Virginica']

    pred_ret = est.predict(features=predict_x)
    # print(list(iter(pred_ret)))  # 纯概率结果，需要进一步解析
    # [{'class_ids': array([0], dtype=int64),
    #   'probabilities': array([9.9774504e-01, 2.2550498e-03, 4.1978092e-09], dtype=float32),
    #   'logits': array([10.6607685, 4.568443, -8.625678], dtype=float32)},
    #  ...]

    for pred_dict, expec in zip(pred_ret, expected_y):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(['Setosa', 'Versicolor', 'Virginica'][class_id],
                              100 * probability, expec))

    """
    Prediction is "Setosa" (99.6%), expected "Setosa"

    Prediction is "Versicolor" (99.7%), expected "Versicolor"
    
    Prediction is "Virginica" (93.5%), expected "Virginica"
    """
