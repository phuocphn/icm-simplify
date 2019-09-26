import tensorflow as tf

class A3C(object):
    def __init__(self, env, worker_task_index, sess=None):
        # we will definite network and all necessary operations in here.

        # define target network in parameter server (`target (global) network weights` and `global step`)
        with tf.device(device_name_or_function=tf.train.replica_device_setter(
                ps_tasks=1, ps_device="/job:ps",
                worker_device="/job:worker/task:{}/cpu:0".format(worker_task_index))):
            with tf.variable_scope("global", reuse=None):
                self.global_network = CNNLSTMPolicy(state_shape = env.observation_space.shape, num_action=env.action_space.n)
                self.global_step = tf.get_variable(name="global_step",
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                #self.global_action_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)


        # define local network in local worker (`local network weights` and `local step`)
        with tf.device(device_name_or_function="/job:worker/task:{}/cpu:0".format(worker_task_index)):
            with tf.variable_scope("local", reuse=None):
                self.local_network =  CNNLSTMPolicy(state_shape = env.observation_space.shape, num_action=env.action_space.n)
                self.local_step = self.global_step
                #self.local_action_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)

    def train(self, sess):
        pass