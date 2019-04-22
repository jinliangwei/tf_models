import tensorflow as tf

global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

with tf.device('/device:GPU:0'):
    X = tf.Variable(tf.random_uniform([5120, 1024]))
    Y = tf.Variable(tf.random_uniform([1024, 5120]))
    Z = tf.matmul(X, Y)

with tf.device('/cpu:0'):
    b = tf.Variable(tf.random_uniform([5120]))
    A = tf.add(Z, b)
    mean = tf.summary.scalar("mean", tf.reduce_mean(A))

init = tf.global_variables_initializer()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                            output_partition_graphs=True)
run_metadata = tf.RunMetadata()

merged = tf.summary.merge_all()

profiler_hook = tf.train.ProfilerHook(save_steps=1,
                                      output_dir="test/logs",
                                      show_memory=True)

with tf.train.MonitoredSession(hooks=[profiler_hook],
                               session_creator=tf.train.ChiefSessionCreator(
                                    config=tf.ConfigProto(log_device_placement=True,
                                                          graph_options=tf.GraphOptions(build_cost_model=1)))) as sess:
    for step in range(1, 3):
        sess.run([A, merged], run_metadata=run_metadata, options=run_options)
        run_metadata_str = run_metadata.SerializeToString()
        with open("run_metadata.dat."+str(step), "wb") as metadata_fobj:
            metadata_fobj.write(run_metadata_str)

cost_graph = run_metadata.cost_graph
print("cost graph, number of nodes = ", len(cost_graph.node))

for partition_graph_def in run_metadata.partition_graphs:
    print(partition_graph_def)
