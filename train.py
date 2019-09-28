import tensorflow as tf
import argparse
from a3c import A3C

# Get user-provided parameters from args
parser = argparse.ArgumentParser()
parser.add_argument('--job_name', default='worker', type=str, help='Job name' )
parser.add_argument('--task_index', default=0, type=int, help='Task Index' )
parser.add_argument('--pretrain_path', default=None, type=str, help='Checkpoint dir (generally ..../train/) to load from.' )
args = parser.parse_args()

JOB_NAME, TASK_INDEX, PRETRAIN_MODEL_PATH = [args.job_name, args.task_index, args.pretrain_path]

# Constants and hyper-parameters
LOG_DIR = '/tmp/doom'
ENV_ID = 'doom'
NUM_WORKERS = 20
TOTAL_TRAINING_STEP = 100           # this is total step and is used for all workers.

cluster = tf.train.ClusterSpec({"ps": "localhost:12200", "worker": ["localhost:12300", "localhost:12301"]})
if JOB_NAME == 'ps':
    server = tf.train.Server(server_or_cluster_def=cluster, job_name=JOB_NAME, task_index=0,
                             config=tf.ConfigProto(device_filters=["/job:ps"]))
    print ("Parameter server is starting...")
    server.join()

if JOB_NAME == 'worker':
    # Create server obj to get managed_session, and then train agent.
    server = tf.train.Server(server_or_cluster_def=cluster, job_name=JOB_NAME, task_index=TASK_INDEX,
                             config=tf.ConfigProto(intra_op_parellelism=1, inter_op_parallelism_threads=2))

    env = None
    trainer = A3C(env=env, worker_task_index=TASK_INDEX) #specify which machine (worker) will be used to train agent.

    # Variables with the name starts with `local...` will not be saved in the checkpoints
    # only save target network and related variables.
    selected_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]
    selected_variables_init_op = tf.variables_initializer(selected_variables)

    saver = tf.train.Saver(var_list=selected_variables)
    supervisor = tf.train.Supervisor(is_chief=(JOB_NAME=='worker' and TASK_INDEX==0),
                             logdir=LOG_DIR,
                             saver=saver,
                             init_op=selected_variables_init_op,
                             ready_op=tf.report_uninitialized_variables(selected_variables),
                             global_step=trainer.global_step,
                             save_model_secs=30 # Number of seconds between the creation of model checkpoints. Defaults to 600 seconds. Pass 0 to disable checkpoints.
                            )
    with supervisor.managed_session(master=server.target,
                                    config=tf.ConfigProto(device_filters=["/job:ps", f"/job:worker/task:{TASK_INDEX}/cpu:0"])) as sess:

        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(PRETRAIN_MODEL_PATH))
        sess.run(tf.global_variables_initializer())
        sess.run(trainer.sync_weight_from_target_network)
        current_training_step =  sess.run(trainer.training_step)    # training_step is put in parameter server.
        print (f"Worker: {JOB_NAME + ':'+ TASK_INDEX } in training step: {current_training_step}")

        while not supervisor.should_stop() and current_training_step < TOTAL_TRAINING_STEP:
            trainer.train(sess=sess)
            current_training_step = sess.run(trainer.training_step)

    # Ask for all the services to stop.
    supervisor.stop()
    print ("Finished !")