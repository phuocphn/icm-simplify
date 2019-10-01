import tensorflow as tf
from extractors import CNNLSTMPolicy
import scipy.signal
import numpy as np
import skimage
from tqdm import tqdm

class A3C(object):
    def __init__(self, env, worker_task_index, sess=None):
        self.env = env
        self.sess = sess
        self.is_chief = (worker_task_index==0)

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
        self.train_op = tf.group(optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step.assign_add(1))) #NOTE: 1 or shape[0]

        # copy weights from the parameter server to the local model.
        sync_assigns = [local_var.assign(global_var) for local_var, global_var in zip(
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="local"),
            tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
        )]
        self.sync_weights_op = tf.group(*sync_assigns)

    def preprocess(self, img, resolution=(84, 84)):
        return np.asarray(skimage.transform.resize(img, resolution).astype(np.float32))

    def train(self, sess):
        self.sess = sess

        # sync weights from global target network
        self.sess.run(self.sync_weights_op)

        current_state = self.preprocess(self.env.reset())
        lengths = 0
        rewards = 0
        values = 0

        # generate batch of episodes
        episode_rollout = EpisodeRollout()
        should_bootstrap = True
        for _ in range(10000):
            action, value =self.sess.run([self.local_network.actions, self.local_network.value_function],
                             feed_dict = {self.local_network.inputs: [current_state]}
                            )

            value = value[0][0]
            next_state, reward, terminal, info = self.env.step(action.argmax())
            next_state = self.preprocess(next_state)

            self.env.render()

            episode_rollout.add(next_state, action, reward, value, terminal)
            rewards += reward
            lengths += 1
            values += value

            current_state = next_state
            if terminal or lengths > self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'):
                terminal = True
                current_state = self.env.reset()
                lengths = 0
                rewards = 0
                should_bootstrap = False
                break

        # if this loop is ended because of out of index and [not terminal state or time-limit max_episode_steps]
        if should_bootstrap:
            bootstrap_value = self.sess.run(self.local_network.value_function, feed_dict={self.local_network.inputs: [current_state]}) [0]
            episode_rollout.update_bootstrap_value(bootstrap_value)

        [batch_states, batch_actions, batch_advantages, batch_rewards, terminal] = episode_rollout.get_training_batch()

        if not terminal:
            print ("ingore ....." * 100)
            return

        if self.is_chief:
            pass

        fetches = [self.train_op, self.global_step]
        feed_dict = {
            self.local_network.inputs: batch_states,
            self.advantages: batch_advantages,
            self.actions: batch_actions,
            self.rewards: batch_rewards,
        }
        fetched = self.sess.run(fetches, feed_dict)
        if terminal:
            print (f"Global step counter: {fetched[-1]}")
        print ("Train ...")

class EpisodeRollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.terminal = False
        self.bootstrap_value = 0.0

    def add(self, state, action, reward, value, terminal):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

    def extend(self, other_history):
        assert  self.terminal == False
        self.states.extend(other_history.states)
        self.actions.extend(other_history.actions)
        self.rewards.extend(other_history.rewards)
        self.values.extend(other_history.values)
        self.terminal = other_history.terminal

    def update_bootstrap_value(self, value):
        self.bootstrap_value = value

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def get_training_batch(self):
        batch_states = np.asarray(self.states)
        batch_actions = np.asarray(self.actions)

        # collecting target for value network
        #clip reward
        rewards = np.asarray(self.rewards)
        rewards = np.clip(rewards, -1, 1)
        batch_rewards = self.discount(rewards + [self.bootstrap_value], gamma=0.99)[:-1]

        #collecting target for policy network
        value_predictions = np.asarray(self.values + [self.bootstrap_value])
        delta_t = rewards + 0.99 * value_predictions[1:] - value_predictions[:-1]
        batch_advantages = self.discount(delta_t, 0.99 * 1.0)

        return [batch_states, batch_actions, batch_advantages, batch_rewards, self.terminal]