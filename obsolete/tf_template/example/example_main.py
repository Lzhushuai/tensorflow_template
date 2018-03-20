import os
import huaytools as hy
import tensorflow as tf
from huaytools.tf_template.example.example_model import ExampleModel, Config
from huaytools.tf_template.utils.data_iris import *

if __name__ == '__main__':
    config = Config('ex')
    config.ckpt_dir = hy.maybe_mkdirs(os.path.join("D:/tmp/log", config.name, "checkpoint/"))
    config.n_units = [10, 10]
    config.n_class = 3

    fcs = [
        tf.feature_column.numeric_column('SepalLength'),
        tf.feature_column.numeric_column('SepalWidth'),
        tf.feature_column.numeric_column('PetalLength'),
        tf.feature_column.numeric_column('PetalWidth')
    ]

    ds_iter = get_train_ds_iter()

    model = ExampleModel(config, feature_columns=fcs)

    with tf.Session() as sess:
        assert model.graph == tf.get_default_graph()
        assert sess.graph == tf.get_default_graph()

        model.load(sess)

        model.train(sess, ds_iter)

        # for i, obj in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
        #     print(i, obj)

        # f, l = ds_iter.get_next()
        # while True:
        #     try:
        #         ff = sess.run(f)
        #         print(ff)
        #     except tf.errors.OutOfRangeError:
        #         break

        from tensorflow.python.tools import inspect_checkpoint as chkp

        latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir)

        # print all tensors in checkpoint file
        chkp.print_tensors_in_checkpoint_file(file_name=latest_ckpt,
                                              tensor_name='', all_tensors=True, all_tensor_names=True)

        tf.estimator