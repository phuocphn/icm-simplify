import tensorflow as tf
import numpy as np

class CNNLSTMPolicy(object):
    """
    Feature extractor: [None, num_features ] ~~> [None, 256]
    """
    def __init__(self, state_shape, num_action):
        """

        :param state_shape:
        :param num_action:
        """

        # https://github.com/mwydmuch/ViZDoom/blob/b50fcd26ffeebb07d9527c8b951976907ef2acfe/examples/python/learning_tensorflow.py
        self.input = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input")

        conv_1 = tf.contrib.layers.convolution2d(self.input, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.elu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        conv_2 = tf.contrib.layers.convolution2d(conv_1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.elu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        conv_3 = tf.contrib.layers.convolution2d(conv_2, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.elu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        conv_4 = tf.contrib.layers.convolution2d(conv_3, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.elu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))

        self.output = tf.reshape(conv_4, [-1, np.prod(conv_4.get_shape().as_list()[1:])])
        self.logits = tf.contrib.layers.fully_connected(self.output, num_outputs=num_action, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.0))



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