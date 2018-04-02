# Tensorflow Template

A deep learning template with tensorflow and it will help you to change just the core part of model every time you start a new tensorflow project, 
of which the idea is from another project [MrGemy95/Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template).
The proj can work well, but it separates the model, train and summary to 3 parts which makes the process more readable but complex. 
And it seems to lack the interface of evaluate and predict.

This proj merges them into a single class and adds more template code. 
The interface of the base class refer to the TF API [tf.estimator.Estimator](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator).
Actually, this proj can not work better than it. It just makes the process more clarity for you, and you can freely change them.

## Table Of Contents

<!-- TOC -->

- [Quick Start](#quick-start)
- [Examples](#examples)

<!-- /TOC -->

## Quick Start

A new version is coming. If you are a tf newer, I recommend you use the original version [v1](./v1) first 
which most use the low-level TF APIs. It can help you learn more about the Tensorflow.

The example is about the iris classification question. You can find the data and code at [examples/iris](./examples/iris)

### 1. Create a model class which inherit the [BaseModel](tensorflow_template/base/base_model.py)

```python
from tensorflow_template.base.base_model import BasicModel

class IrisModel(BasicModel):

    def _init_model(self, features, labels, mode):
        pass
```

All you need to do is fill the model function with some necessary and custom [tf ops](https://www.tensorflow.org/versions/master/api_docs/python/tf/Operation).

### 2. Build the graph
    
The basic part:
- the neural net
- the output for predict (such as logits, argmax(logits), softmax(logits))
- the loss
- the train_op

others:
- the metrics_op (such as `tf.metrics.accuracy`)
  
For reuse the same model function in different mode(train/eval/predict), 
you have to make it return different ops with different mode 
which may a little complex than the v1 version, but it is worth.
 
```python
import tensorflow as tf

def _init_graph(self, features, labels, mode):
    # 1. define the neural net
    # for units in self.config.n_units:
    #     net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    net = tf.layers.dense(features, units=self.config.n_units[0], activation=tf.nn.relu)
    net = tf.layers.dense(net, units=self.config.n_units[1], activation=tf.nn.relu)

    # the output
    self._logits = tf.layers.dense(net, self.config.n_class, activation=None)
    self._prediction = tf.argmax(self._logits, axis=1)

    if mode == self.ModeKeys.PREDICT:
        self._probabilities = tf.nn.softmax(self._logits)
        # return [self._prediction, self._probabilities]
        # It's better to use dict
        return {'prediction': self._prediction,
                'probabilities': self._probabilities}
                
    # 2. define the loss
    self._loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)
    # self.add_log_tensor('loss', self._loss)

    # 3. define the metrics if needed
    _, self._accuracy = tf.metrics.accuracy(labels=labels,
                                            predictions=self._prediction, name='acc_op')
    # self.add_log_tensor('acc', self._accuracy)

    self.add_log_tensors({'loss': self._loss,
                          'acc': self._accuracy})

    if mode == self.ModeKeys.EVALUATE:
        return [self._loss, self._accuracy]
    
    # 4. define the train_op when the mode is TRAIN
    if mode == self.ModeKeys.TRAIN:
        self._optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self._train_op = self._optimizer.minimize(self._loss, global_step=tf.train.get_global_step())

        return [self._train_op, self._loss, self._accuracy]
```
    
### 3. Set your Config

Generally, I have help you set almost the config args with a default value.
If you don'y know what these arg means yet, just run it.

Note, most of these args are not the hyperparameters in deep learning except the learning rate.
They are just the boresome parameters of different kinds of the APIs from tensorflow.

You can easily modify the predefine args or add new args with the Config class.

Here `Config` object is a subclass of Bunch object. If you want to use it, just `pip install bunch`.

```python
from tensorflow_template.base.base_model import BaseConfig

def get_config():
    config = BaseConfig('ex')
    
    config.n_epoch_train = 100  # modify the predefine args
    config.aaa = 0.1  # add new args
    
    return config
```
    
### 4. Run it
    
```python
if __name__ == '__main__':

    config = get_config()
    ds_train, ds_eval, ds_pred = get_datasets()
    
    model = IrisModel(config)
    
    model.train(ds_train)
    model.evaluate(ds_eval)
    ret = model.predict(ds_pred)
    
    for i in ret:
        print(i)
```

The dataset is a instance of `tf.data.Dataset`, which is few easy-to-use APIs of tensorflow.
If you don't know how to use it, learning it from the official tutorials [Datasets Quick Start](https://www.tensorflow.org/versions/master/get_started/datasets_quickstart).

Or see the **data_helper.py** of each [examples](./examples).

## Examples

- iris classification 
  - [[code](./examples/iris)] with tf_template
  - ref: [TF Tutorials](https://www.tensorflow.org/get_started/get_started_for_beginners)
- mnist classification 
  - [[code](./examples/mnist)] with tf_template
  - ref: [TF Tutorials](https://www.tensorflow.org/tutorials/layers)
- cnn-text-classification [TODO]
  - The original [paper](http://arxiv.org/abs/1408.5882) and [github](https://github.com/yoonkim/CNN_sentence)
    
## Contributing

If you have new requests, please add new issue.

If you use the template to build some model. Welcome it to the examples.