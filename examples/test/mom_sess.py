import numpy as np
import tensorflow as tf
import huaytools as hy
from tensorflow_template.base.base_config import BaseConfig

hy.set_logging_basic_config()

config = BaseConfig('test')

config.save_ckpt_steps = 1
config.save_sum_steps_eval = 1
config.log_n_steps_train = 1
config.sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))


def get_dataset():
    n = 100
    features = np.random.random((n, 10))
    labels = np.array([0, 1, 2, 3, 4] * (n // 5))

    return tf.data.Dataset.from_tensor_slices((features, labels))


dataset = get_dataset()
dataset = dataset.shuffle(config.shuffle_buffer_size).batch(config.n_batch_train).repeat(5)

hooks = []
hooks.extend([tf.train.CheckpointSaverHook(config.ckpt_dir, save_steps=config.save_ckpt_steps)])

with tf.Graph().as_default() as g:
    ds_iter = dataset.make_one_shot_iterator()

    features, labels = ds_iter.get_next()

    global_step = tf.train.create_global_step()

    net = tf.layers.dense(features, units=32, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 5, activation=None)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    summary_loss = tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    hooks.extend([tf.train.LoggingTensorHook({'loss': loss, 'step': global_step},
                                             every_n_iter=config.log_n_steps_train),
                  tf.train.SummarySaverHook(save_steps=config.save_sum_steps_eval,
                                            output_dir=config.summary_dir,
                                            summary_op=summary_loss)])

    with tf.train.MonitoredTrainingSession(  # session_creator=tf.train.ChiefSessionCreator(config=config.sess_config),
            hooks=hooks) as sess:
        while not sess.should_stop():
            _, loss_val = sess.run([train_op, loss])
