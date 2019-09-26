import tensorflow as tf
import numpy as np

# Create some wrappers for simplicity
def conv2d(variable_namescope, input, strides=[2, 2], filter_size=[3, 3], num_filters=32):
    # Conv2D wrapper, with bias and relu activation
    with tf.variable_scope(variable_namescope):
        W = tf.get_variable("W", shape=[filter_size[0], filter_size[1], int(input.get_shape()[3]), num_filters],
                            dtype=tf.float32,
                            # https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
                                                                                       uniform=True, seed=None,
                                                                                       dtype=tf.float32))
        b = tf.get_variable("b", shape=[1, 1, 1, num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.elu(tf.nn.conv2d(input, filter=W, strides=[1, strides[0], strides[1], 1], padding="SAME") + b)


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
            __input = conv2d(variable_namescope="layer_%s" % idx, input=__input, strides=[2,2], filter_size=[3,3], num_filters=32)
        self.output = tf.reshape(__input, [-1, np.prod(__input.get_shape().as_list()[1:])])

        # Last fully connected layer (value function).
        w = tf.get_variable("value_function/w", [self.output.get_shape()[1], num_action], initializer=normalized_columns_initializer(1.0))
        b = tf.get_variable("value_function/b", [num_action], initializer=tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.output, w) + b



"""
def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)
"""