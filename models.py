import tensorflow as tf
import numpy as np

# Create some wrappers for simplicity
def conv2d(name, input, strides=[2, 2], filter_size=[3, 3], num_filters=32):
    # Conv2D wrapper, with bias and relu activation
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[filter_size[0], filter_size[1], int(input.shape()[3]), num_filters],
                            dtype=tf.float32,
                            # https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
                                                                                       uniform=True, seed=None,
                                                                                       dtype=tf.float32))
        b = tf.get_variable("b", shape=[1, 1, 1, 32], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.elu(
            tf.nn.bias_add(tf.nn.conv2d(input, filter=W, strides=[1, strides[0], strides[1], 1], padding="SAME"), b))


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class CNNLSTMPolicy(object):
    """
    Feature extractor: [None, num_features ] ~~> [None, 256]
    """
    def __init__(self, state_shape, num_action):
        """

        :param state_shape:
        :param num_action:
        """
        self.input = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input")

        # 4 convolution layer stack together.
        __input = self.input
        for idx in range(4):
            __input = conv2d(name=f"layer_{idx}", input=__input, strides=[2,2], filter_size=[3,3], num_filters=32)
        self.output = tf.reshape(__input, [-1, np.prod(__input.get_shape().as_list()[1:])])

        # Last fully connected layer (value function).
        w = tf.get_variable("value_function/w", [__input.get_shape()[1], num_action], initializer=normalized_columns_initializer(1.0))
        b = tf.get_variable("value_function/b", [num_action], initializer=tf.constant_initializer(0.0))
        self.logits = tf.matmul(__input, w) + b
