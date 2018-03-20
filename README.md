# Tensorflow Template

A deep learning template with tensorflow and it will will help you to change just the core part of model every time you start a new tensorflow project, 
of which the idea is from another project [MrGemy95/Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template).
The proj can work well, but it separates the model, train and summary to 3 parts which makes the process more readable but complex. 
And it seems to lack the interface of evaluate and predict.

This proj merges them into a single class and adds more template code. The interface of the base class refer to the TF API [tf.estimator.Estimator](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator).
Actually, this proj can not work better than it. It just makes the process more clarity for you, and you can freely change them.


## Quick Start

