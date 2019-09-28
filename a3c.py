import tensorflow as tf
from extractors import CNNLSTMPolicy

class A3C(object):
    def __init__(self, env, worker_task_index, sess=None):
        # we will definite network and all necessary operations in here.

        # define target network in parameter server (`target (global) network weights` and `global step`)
        with tf.device(device_name_or_function=tf.train.replica_device_setter(
                ps_tasks=1, ps_device="/job:ps",
                worker_device="/job:worker/task:{}/cpu:0".format(worker_task_index))):
            with tf.variable_scope("global", reuse=None):
                self.global_network = CNNLSTMPolicy(state_shape = [84, 84, 3], num_action=5) #NOTE: get state_shape from env.observation_space later.
                self.global_step = tf.get_variable(name="global_step",
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                #NOTE: #ICM,  we will implement this later
                #self.global_prediction_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)


        # define local network in local worker (`local network weights` and `local step`)
        with tf.device(device_name_or_function="/job:worker/task:{}/cpu:0".format(worker_task_index)):
            with tf.variable_scope("local", reuse=None):
                self.local_network =  CNNLSTMPolicy(state_shape = [84, 84, 3], num_action=5)
                self.local_step = self.global_step
                #NOTE: #ICM,  we will implement this later
                #self.local_prediction_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)


        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="actions") #NOTE: get shape from env.action_space.n later.
        self.advantages = tf.placeholder(dtype=tf.float32, shape=[None], name="advantages")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")

        # https://discuss.pytorch.org/t/what-is-the-difference-between-log-softmax-and-softmax/11801
        probs = tf.nn.softmax(self.local_network.logits)

        policy_loss =  -tf.reduce_mean(input_tensor= tf.reduce_sum(tf.log(probs) * self.actions, axis=1) * self.advantages) #scalar value
        value_function_loss = 0.5 * tf.reduce_mean(tf.square(self.local_network.value_function - self.rewards))
        entropy_loss = -tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs), axis=1)) #element-wise multiplication
        self.loss = policy_loss + 0.5 * value_function_loss - entropy_loss * 0.01

        gradients = tf.gradients(self.loss, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="local"))
        # NOTE: #ICM,  we will implement this later
        #self.prediction_network_gradients =  0.01 * self.local_prediction_network.

        gradients, gradient_norms = tf.clip_by_global_norm(t_list=gradients,clip_norm=40.0)
        grads_and_vars = list(zip(gradients, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")))

        optimizer = tf.train.AdamOptimizer(learning_rate=float(1e-4))
        self.train_op = tf.group(optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step.assign_add(1)))

        # copy weights from the parameter server to the local model.
        sync_assigns = [local_var.assign(global_var) for local_var, global_var in zip(
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="local"),
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
        )]
        self.sync_weights_op = tf.group(*sync_assigns)

    def train(self, sess):
        pass