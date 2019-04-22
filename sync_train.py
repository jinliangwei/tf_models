import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

FLAGS = None

# Parameters
training_steps = 25
batch_size = 25
display_step = 10

log_dir = "train_logs/"

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  num_workers = len(worker_hosts)

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      x = tf.placeholder(tf.float32, [None, 784])
      y = tf.placeholder(tf.float32, [None, 10])

      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))

      pred = tf.nn.softmax(tf.matmul(x, W) + b)
      loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
      global_step = tf.contrib.framework.get_or_create_global_step()

      opt = tf.train.AdagradOptimizer(0.01)
      opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_workers,
                                     total_num_replicas=num_workers)
      train_op = opt.minimize(loss, global_step=global_step)
    
    is_chief = (FLAGS.task_index == 0)

    # The StopAtStepHook handles stopping after running given steps.
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                output_partition_graphs=True)
    run_metadata = tf.RunMetadata()
    merged = tf.summary.merge_all()
    profiler_hook = tf.train.ProfilerHook(save_steps=display_step,
                                          output_dir=log_dir+"profiler",
                                          show_memory=True)
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    hooks=[profiler_hook, tf.train.StopAtStepHook(last_step=100), sync_replicas_hook]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=log_dir,
                                           hooks=hooks) as mon_sess:
      step = 0
      while not mon_sess.should_stop():
        step += 1
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        mon_sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys},
                     run_metadata=run_metadata, options=run_options)
        if step % display_step == 0:
            run_metadata_str = run_metadata.SerializeToString()
            with open(log_dir+"run_metadata.dat."+str(step), "wb") as metadata_fobj:
                metadata_fobj.write(run_metadata_str)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
