import tensorflow as tf
from policys import CNNLSTMPolicy, StateActionPredictor
import scipy.signal
import numpy as np
import skimage

class A3C(object):
    def __init__(self, env, worker_task_index, sess=None):
        self.env = env
        self.sess = sess
        self.is_chief = (worker_task_index==0)
        self.worker_task_index = worker_task_index

        # we will definite network and all necessary operations in here.

        # define target network in parameter server (`target (global) network weights` and `global step`)
        with tf.device(device_name_or_function=tf.train.replica_device_setter(
                ps_tasks=1, ps_device="/job:ps",
                worker_device="/job:worker/task:{}/cpu:0".format(worker_task_index))):
            with tf.variable_scope("global", reuse=None):
                self.global_network = CNNLSTMPolicy(state_shape = [160, 120, 3], num_action=env.action_space.n) #NOTE: get state_shape from env.observation_space later.
                self.global_step = tf.get_variable(name="global_step",
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                with tf.variable_scope("predictor"):
                    self.global_state_action_predictor = StateActionPredictor(ob_space=[160, 120, 3], ac_space=env.action_space.n)
                #NOTE: #ICM,  we will implement this later
                #self.global_prediction_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)


        # define local network in local worker (`local network weights` and `local step`)
        with tf.device(device_name_or_function="/job:worker/task:{}/cpu:0".format(worker_task_index)):
            with tf.variable_scope("local", reuse=None):
                self.local_network  =  CNNLSTMPolicy(state_shape = [160, 120, 3], num_action=env.action_space.n)
                self.local_network.global_step = self.global_step
                with tf.variable_scope("predictor"):
                    self.local_state_action_predictor = StateActionPredictor(ob_space=[160, 120, 3], ac_space=env.action_space.n)
                #NOTE: #ICM,  we will implement this later
                #self.local_prediction_network = StateActionPredictor(state_shape = env.observation_space.shape, num_action=env.action_space.n)


            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name="actions") #NOTE: get shape from env.action_space.n later.
            self.advantages = tf.placeholder(dtype=tf.float32, shape=[None], name="advantages")
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")

            # https://discuss.pytorch.org/t/what-is-the-difference-between-log-softmax-and-softmax/11801
            probs = tf.nn.softmax(self.local_network.logits)

            policy_loss =  -tf.reduce_mean(input_tensor= tf.reduce_sum(tf.log(probs) * self.actions, axis=1) * self.advantages) #scalar value
            value_function_loss = 0.5 * tf.reduce_mean(tf.square(self.local_network.value_function - self.rewards))
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs), axis=1)) #element-wise multiplication
            self.loss = policy_loss + 0.5 * value_function_loss - entropy_loss * 0.01

            gradients = tf.gradients(self.loss, self.local_network.var_list)

            # ICM
            self.predict_loss = 10.0 * (self.local_state_action_predictor.invese_loss * (1-0.2) + self.local_state_action_predictor.forward_loss * 0.2)
            #self.predict_loss = 10.0 * (self.local_state_action_predictor.invese_loss )

            predict_gradients = tf.gradients(self.predict_loss * 20.0, self.local_state_action_predictor.var_list)
            print ("$"*100)
            print (self.local_state_action_predictor.var_list)
            print ("*"*100)
            print (predict_gradients)
            #exit()
            tf.summary.scalar("model/policy_loss", policy_loss )
            tf.summary.scalar("model/value_loss", value_function_loss )
            tf.summary.scalar("model/entropy", entropy_loss)
            tf.summary.scalar("model/reward_mean", tf.math.reduce_mean(self.rewards))
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(gradients))
            tf.summary.scalar("model/variable_global_norm", tf.global_norm(self.local_network.var_list))
            if True: # use ICM
                tf.summary.scalar("model/inverse_loss", self.local_state_action_predictor.invese_loss)
                tf.summary.scalar("model/forward_loss", self.local_state_action_predictor.forward_loss)
                tf.summary.scalar("model/predgrad_global_norm", tf.global_norm(predict_gradients))
                tf.summary.scalar("model/predvar_global_norm", tf.global_norm(self.local_state_action_predictor.var_list))

            self.summary_op = tf.summary.merge_all()

            gradients, gradient_norms = tf.clip_by_global_norm(gradients,clip_norm=40.0)
            grads_and_vars = list(zip(gradients, self.global_network.var_list))
            if True: # use ICM
                print ("before   ---- predict_gradients: " )
                print ("*"*100)
                print (predict_gradients)
                print ("*"*100)


                predict_gradients, _ = tf.clip_by_global_norm(predict_gradients, clip_norm=40.0)
                predict_gradients_and_vars = list(zip(predict_gradients, self.global_state_action_predictor.var_list))
                print ("predict_gradients: " )
                print ("*"*100)
                print (predict_gradients)
                print ("*"*100)
                print ("predict_gradients_and_vars: " )
                print ("*"*100)
                print (predict_gradients_and_vars)
                print ("*"*100)

                print (predict_gradients_and_vars)
                print ("#"*1000)
                print (grads_and_vars)
                print ("#"*1000)
                #ex_delta_t = tf.reduce_mean(tf.concat([tf.reshape(g, [-1]) for g in tf.gradients(t
                grads_and_vars = grads_and_vars + predict_gradients_and_vars
                print ("*"*100)
                print (grads_and_vars)
                #exit()

            optimizer = tf.train.AdamOptimizer(learning_rate=float(1e-4))
            self.train_op = tf.group(optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                                               global_step=self.global_step.assign_add( 1)))

            # copy weights from the parameter server to the local model.
            sync_assigns = [local_var.assign(global_var) for local_var, global_var in zip(
                self.local_network.var_list,
                self.global_network.var_list
            )]

            if True: # use ICM
                sync_assigns += [local_var.assign(global_var) for local_var, global_var in zip(
                    self.local_state_action_predictor.var_list,
                    self.global_state_action_predictor.var_list
                )]
            self.sync_weights_op = tf.group(*sync_assigns)

            self.summary_writer = None
            self.local_step = 0

    def preprocess(self, img, resolution=(160, 120)):
        return np.asarray(skimage.transform.resize(img, resolution).astype(np.float32))

    def provide_context(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer

    def train(self, sess, summary_writer):

        # sync weights from global target network
        self.sess.run(self.sync_weights_op)

        current_state = self.preprocess(self.env.reset())
        rnn_features = self.local_network.state_init


        lengths = 0
        rewards = 0
        values = 0

        if True: # use ICM
            life_bonus = 0
            episode_bonus = 0

        # generate batch of episodes
        episode_rollout = EpisodeRollout()
        should_bootstrap = True
        for _ in range(10000):
            action, value, features =self.sess.run([self.local_network.actions, self.local_network.value_function, self.local_network.state_out],
                             feed_dict = {self.local_network.inputs: [current_state],
                                          self.local_network.state_in[0]: rnn_features[0],
                                          self.local_network.state_in[1]: rnn_features[1]}
                            )
            value = value[0][0]
            next_state, reward, terminal, info = self.env.step(action.argmax())
            next_state = self.preprocess(next_state)

            self.env.render()

            current_tuple = [current_state, action, reward, value, terminal, rnn_features]
            if True: #use ICM
                bonus = self.local_state_action_predictor.predict_bonus(state_1=current_state, state_2=next_state, action_sample=action)
                current_tuple += [bonus, next_state]
                life_bonus += bonus
                episode_bonus += bonus

            episode_rollout.add(*current_tuple)
            rewards += reward
            lengths += 1
            values += value

            current_state = next_state
            rnn_features = features

            if terminal or lengths > self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'):
                terminal = True
                current_state = self.env.reset()
                rnn_features = self.local_network.state_init
                lengths = 0
                rewards = 0
                should_bootstrap = False

                if True: #use ICM
                    life_bonus = 0

                break

        # if this loop is ended because of out of index and [not terminal state or time-limit max_episode_steps]
        if should_bootstrap:
            bootstrap_value = self.sess.run(self.local_network.value_function,
                                            feed_dict={self.local_network.inputs: [current_state],
                                                       self.local_network.state_in[0]: rnn_features[0],
                                                       self.local_network.state_in[1]: rnn_features[1]}) [0]
            episode_rollout.update_bootstrap_value(bootstrap_value)

        [batch_states, batch_actions, batch_advantages, batch_rewards, terminal, batch_features] = episode_rollout.get_training_batch()

        if not terminal:
            print ("*" * 100)
            print ("ignore.")
            return

        should_write_summary = (self.is_chief and self.local_step % 10 == 0)
        if should_write_summary:
            fetches =  [self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.inputs: batch_states,
            self.advantages: batch_advantages,
            self.actions: batch_actions,
            self.rewards: batch_rewards,
            self.local_network.state_in[0]: batch_features[0],
            self.local_network.state_in[1]: batch_features[1],
        }

        if True: #use ICM
            feed_dict[self.local_network.inputs] = batch_states[:-1]
            feed_dict[self.local_state_action_predictor.state_1] = batch_states[:-1]
            feed_dict[self.local_state_action_predictor.state_2] = batch_states[1:]
            feed_dict[self.local_state_action_predictor.action_sample] = batch_actions

        fetched = self.sess.run(fetches, feed_dict=feed_dict)
        self.local_step += 1

        if should_write_summary:
            summary = sess.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, fetched[-1])
            self.summary_writer.flush()
        print (f"*** Worker {self.worker_task_index} at local step: {self.local_step}, reward_mean: {np.mean(batch_rewards)}")

class EpisodeRollout(object):
    def __init__(self, use_icm=True):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.terminal = False
        self.features = []
        self.bootstrap_value = 0.0
        self.use_icm = use_icm
        if use_icm:
            self.bonuses = []
            self.end_state = None

    def add(self, state, action, reward, value, terminal,features, bonus=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.use_icm:
            self.bonuses += [bonus]
            self.end_state = end_state


    def extend(self, other_history):
        assert  self.terminal == False
        self.states.extend(other_history.states)
        self.actions.extend(other_history.actions)
        self.rewards.extend(other_history.rewards)
        self.values.extend(other_history.values)
        self.terminal = other_history.terminal
        self.features.extend(other_history.features)

        if self.use_icm:
            self.bonuses.extend(other_history.bonuses)
            self.end_state = other_history.end_state

    def update_bootstrap_value(self, value):
        self.bootstrap_value = value

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def get_training_batch(self):
        if True: #use ICM
            batch_states = np.asarray(self.states +  [self.end_state])
        else:
            batch_states = np.asarray(self.states)
        batch_actions = np.asarray(self.actions)

        # collecting target for value network
        rewards_plus_v = np.asarray(self.rewards + [self.bootstrap_value])
        if True: #use ICM
            rewards_plus_v += np.asarray(self.bonuses + [0])
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -1.0, 1.0)
        batch_rewards = self.discount(rewards_plus_v, gamma=0.99)[:-1]

        #collecting target for policy network
        rewards = np.asarray(self.rewards)
        if True: #use ICM
            rewards += np.asarray(self.bonuses)
        rewards = np.clip(rewards, -1, 1)
        value_predictions = np.asarray(self.values + [self.bootstrap_value])


        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
        # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
        delta_t = rewards + 0.99 * value_predictions[1:] - value_predictions[:-1]
        batch_advantages = self.discount(delta_t, 0.99 * 1.0)

        # features in get_training_batch:  (624, 2, 1, 256)ip
        # print ("features in get_training_batch: ", np.asarray(self.features).shape)
        return [batch_states, batch_actions, batch_advantages, batch_rewards, self.terminal, self.features[0]]