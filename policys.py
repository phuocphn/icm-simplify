import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

def FeatureExtractor(inputs):
    with tf.variable_scope("conv_1"):
        conv_1 = tf.contrib.layers.convolution2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                                 activation_fn=tf.nn.elu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                 biases_initializer=tf.constant_initializer(0.1))

    with tf.variable_scope("conv_2"):
        conv_2 = tf.contrib.layers.convolution2d(conv_1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                             activation_fn=tf.nn.elu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             biases_initializer=tf.constant_initializer(0.1))

    with tf.variable_scope("conv_3"):
        conv_3 = tf.contrib.layers.convolution2d(conv_2, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                             activation_fn=tf.nn.elu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             biases_initializer=tf.constant_initializer(0.1))

    with tf.variable_scope("conv_4"):
        conv_4 = tf.contrib.layers.convolution2d(conv_3, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                             activation_fn=tf.nn.elu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             biases_initializer=tf.constant_initializer(0.1))

    with tf.variable_scope("output_flatten"):
        output = tf.reshape(conv_4, [-1, np.prod(conv_4.get_shape().as_list()[1:])])

    return output


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

        # conv_1 = tf.contrib.layers.convolution2d(self.inputs, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
        #                                         activation_fn=tf.nn.elu,
        #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                         biases_initializer=tf.constant_initializer(0.1))
        #
        # conv_2 = tf.contrib.layers.convolution2d(conv_1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
        #                                         activation_fn=tf.nn.elu,
        #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                         biases_initializer=tf.constant_initializer(0.1))
        #
        # conv_3 = tf.contrib.layers.convolution2d(conv_2, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
        #                                         activation_fn=tf.nn.elu,
        #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                         biases_initializer=tf.constant_initializer(0.1))
        #
        # conv_4 = tf.contrib.layers.convolution2d(conv_3, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
        #                                         activation_fn=tf.nn.elu,
        #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        #                                         biases_initializer=tf.constant_initializer(0.1))

        # self.output = tf.reshape(conv_4, [-1, np.prod(conv_4.get_shape().as_list()[1:])])
        self.output = FeatureExtractor(self.inputs)
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
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)



class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space):
        self.state_1 = phi1 = tf.placeholder(tf.float32, shape=[None] + list(ob_space))
        self.state_2 = phi2 = tf.placeholder(tf.float32, shape=[None] + list(ob_space))

        self.action_sample = action_sample = tf.placeholder(tf.float32, [None, ac_space])

        phi1 = FeatureExtractor(phi1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            phi2 = FeatureExtractor(phi2)

        # Inverse model: g(phi1, phi2) --> action_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1 )
        g = tf.contrib.layers.fully_connected(g, num_outputs=256, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
        logits = tf.contrib.layers.fully_connected(g, num_outputs=ac_space, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
        action_indexes = tf.argmax(action_sample, axis=1)
        self.invese_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_indexes), name="inverse_loss")
        self.action_inverse_probs = tf.nn.softmax(logits, dim=-1)


        # Forward model: f(phi1, action_sample) --> phi2
        f = tf.concat([phi1, action_sample], 1)
        f = tf.contrib.layers.fully_connected(f, num_outputs=256, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.constant_initializer(0.1))

        f = tf.contrib.layers.fully_connected(f, num_outputs=phi1.get_shape()[1].value, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))
        self.forward_loss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name="forward_loss")
        self.forward_loss *= 288.0

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def predict_action(self, state_1, state_2):
        """
        :return action probability distribution predicted by inverse model.
        :param state_1:
        :param state_2:
        :return:
        """
        sess = tf.get_default_session()
        return sess.run(self.action_inverse_probs, {self.state_1: [state_1], self.state_2: [state_2]})[0, :]

    def predict_bonus(self, state_1, state_2, action_sample):
        """
        :return" bonus predicted by forward model
        :param state_1:
        :param state_2:
        :param action_sample:
        :return:
        """
        sess = tf.get_default_session()
        bonus = sess.run(self.forward_loss, {self.state_1: [state_1], self.state_2: [state_2], self.action_sample: [action_sample] })
        bonus *= 0.01
        return bonus