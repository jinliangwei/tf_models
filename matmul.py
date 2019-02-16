import tensorflow as tf

global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

with tf.device('/device:GPU:0'):
    X = tf.Variable(tf.random_uniform([512, 1024]))
    Y = tf.Variable(tf.random_uniform([1024, 1]))
    Z = tf.matmul(X, Y)

with tf.device('/cpu:0'):
    b = tf.Variable(tf.random_uniform([512]))
    A = tf.add(Z, b)

init = tf.global_variables_initializer()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                            output_partition_graphs=True)
run_metadata = tf.RunMetadata()

profiler_hook = tf.train.ProfilerHook(save_steps=1,
                                      output_dir="test/logs",
                                      show_memory=True)

with tf.train.MonitoredSession(hooks=[profiler_hook],
                               session_creator=tf.train.ChiefSessionCreator(
                                    config=tf.ConfigProto(log_device_placement=True,
                                                          graph_options=tf.GraphOptions(build_cost_model=1)))) as sess:
    sess.run([A], run_metadata=run_metadata, options=run_options)
    
