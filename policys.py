import tensorflow as tf
import tensorflow.contrib.rnn as rnn
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
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="inputs")

        conv_1 = tf.contrib.layers.convolution2d(self.inputs, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
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
        self.output = tf.expand_dims(input=self.output, axis=0)

        lstm_cell = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
        #self.state_size = lstm_cell.state_size
        step_size = tf.shape(self.inputs)[:1]

        c_init = np.zeros(shape=(1, lstm_cell.state_size.c), dtype=np.float32)
        h_init = np.zeros(shape=(1, lstm_cell.state_size.h), dtype=np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(dtype=tf.float32, shape=[1, lstm_cell.state_size.c], name='c_in')
        h_in = tf.placeholder(dtype=tf.float32, shape=[1, lstm_cell.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in,h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, self.output, initial_state=state_in, sequence_length=step_size, time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.outputs = tf.reshape(lstm_outputs, [-1, 256])

        # ACTOR : A policy function, controls how our agent acts.
        self.logits = tf.contrib.layers.fully_connected(self.outputs, num_outputs=num_action, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))

        # CRITIC : A value function, measures how good these actions are.
        self.value_function = tf.contrib.layers.fully_connected(self.outputs, num_outputs=1, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
        print ("*" * 50)
        #print ("Number of trainable_variables: ", len(tf.trainable_variables()))
        trainable_variables = tf.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        print ("Number of trainable_variables: ", len(trainable_variables))
        print ("Current variable_scope: ", tf.get_variable_scope().name)
        print ("Total trainable parameters: ", np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables]))

        # >> > sess.run(tf.multinomial([[8.0, 10.0]], 5))
        # array([[1, 1, 1, 0, 1]])
        # http://docs.w3cub.com/tensorflow~python/tf/multinomial/

        # this operation is used for one example mini-batch only
        # (generating episodes or the inference time).
        self.actions = tf.one_hot(tf.squeeze(input=tf.multinomial(logits=self.logits, num_samples=1), axis=1), depth=num_action, name='one_hot')[0, :]

        for var in trainable_variables:
            print (var)
        print ("*" * 50)
