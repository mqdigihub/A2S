import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def _relu(x):

    return tf.nn.relu(x)

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.truncated_normal_initializer(
                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))

        bias = tf.get_variable('bias', [out_channel], initializer=tf.constant_initializer(0.0))
        # if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
        #     tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
        conv = tf.nn.bias_add(conv, bias)
    return conv

def _fc(x, out_dim, regurizer, name='fc'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [x.shape[1], out_dim],
                        tf.float32, initializer=tf.truncated_normal_initializer(
                            stddev=np.sqrt(1.0/out_dim)))
        regurization = regurizer(w)
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.1))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc, regurization


def _bn(x, is_train, name):

    with tf.variable_scope(name):

        bn = tf.layers.batch_normalization(x, training=is_train)

    return bn

## Other helper functions



