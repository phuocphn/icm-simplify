import tensorflow as tf
import argparse
from a3c import A3C
import gym
import time
import os
import numpy as np

# Get user-provided parameters from args
parser = argparse.ArgumentParser()
parser.add_argument('--job_name', default='worker', type=str, help='Job name' )
parser.add_argument('--task_index', default=0, type=int, help='Task Index' )
parser.add_argument('--pretrain_path', default=None, type=str, help='Checkpoint dir (generally ..../train/) to load from.' )
args = parser.parse_args()

JOB_NAME, TASK_INDEX, PRETRAIN_MODEL_PATH = [args.job_name, args.task_index, args.pretrain_path]

# Constants and hyper-parameters
LOG_DIR = '/tmp/mspacman-v0'
ENV_ID = 'MsPacman-v0'
NUM_WORKERS = 20
TOTAL_TRAINING_STEP = 1000000           # this is total step and is used for all workers.

env = gym.make('MsPacman-v0')
print ("*" * 50)
#(0 = center, 1 = up, 2 = right, 3 = left, 4 = down, 5 = upper-right, 6 = upper-left, 7 = lower-right, 8 = lower-left)
print ("Observation space: ", env.observation_space)
print ("Action space: ", env.action_space)
print ("env.spec.timestep_limit: ", env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
print ("*" * 50)
time.sleep(5)

cluster = tf.train.ClusterSpec({"ps": ["localhost:12200"], "worker": ["localhost:12300", "localhost:12301", "localhost:12302", "localhost:12303"]})
if JOB_NAME == 'ps':
    #os.system("kill -9 $( lsof -i:12200 -t ) > /dev/null 2>&1")
    server = tf.train.Server(server_or_cluster_def=cluster, job_name=JOB_NAME, task_index=0,
                             config=tf.ConfigProto(device_filters=["/job:ps"]))
    print ("Parameter server is starting...")
    server.join()

if JOB_NAME == 'worker':
    # Create server obj to get managed_session, and then train agent.
    #os.system("kill -9 $( lsof -i:12300-12301 -t ) > /dev/null 2>&1")
    server = tf.train.Server(server_or_cluster_def=cluster, job_name=JOB_NAME, task_index=TASK_INDEX,
                             config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))

    env = gym.make('MsPacman-v0')
    trainer = A3C(env=env, worker_task_index=TASK_INDEX) #specify which machine (worker) will be used to train agent.

    # Variables with the name starts with `local...` will not be saved in the checkpoints
    # only save target network and related variables.
    selected_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]
    selected_variables_init_op = tf.variables_initializer(selected_variables)

    saver = tf.train.Saver(var_list=selected_variables)
    summary_writer = tf.summary.FileWriter(LOG_DIR + "__%d" % TASK_INDEX)

    supervisor = tf.train.Supervisor(is_chief=(JOB_NAME=='worker' and TASK_INDEX==0),
                             logdir=LOG_DIR,
                             saver=saver,
                             init_op=selected_variables_init_op,
                             summary_writer=summary_writer,
                             summary_op=None,
                             ready_op=tf.report_uninitialized_variables(selected_variables),
                             global_step=trainer.global_step,
                             save_model_secs=30 # Number of seconds between the creation of model checkpoints. Defaults to 600 seconds. Pass 0 to disable checkpoints.
                            )
    with supervisor.managed_session(master=server.target,
                                    config=tf.ConfigProto(device_filters=["/job:ps", f"/job:worker/task:{TASK_INDEX}/cpu:0"])) as sess, sess.as_default():

        if PRETRAIN_MODEL_PATH:
            saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(PRETRAIN_MODEL_PATH))

        sess.run(trainer.sync_weights_op)
        trainer.provide_context(sess=sess, summary_writer=summary_writer)
        global_step =  sess.run(trainer.global_step)    # training_step is put in parameter server.
        print (f"Worker: {JOB_NAME + ':'+ str(TASK_INDEX) } start training at global step: {str(global_step)}")

        while not supervisor.should_stop() and global_step < TOTAL_TRAINING_STEP:
            trainer.train(sess=sess, summary_writer=summary_writer)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    supervisor.stop()
    print ("Finished !")