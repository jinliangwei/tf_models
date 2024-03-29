""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import sys
import tensorflow as tf
from google.protobuf import text_format

from tensorflow.python.client import timeline

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print("read dataset done!", file = sys.stderr)

# Training Parameters
learning_rate = 0.001
num_steps = 1
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input], name="X")
Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name = "wc1"),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name = "wc2"),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name = "wd1"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]), name = "weights_out")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name = "bc1"),
    'bc2': tf.Variable(tf.random_normal([64]), name = "bc2"),
    'bd1': tf.Variable(tf.random_normal([1024]), name = "bd1"),
    'out': tf.Variable(tf.random_normal([num_classes]), name = "bias_out")
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

train_op = optimizer.minimize(loss_op, global_step=global_step_tensor)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                            output_partition_graphs=True)
print(run_options)
run_metadata = tf.RunMetadata()

merged = tf.summary.merge_all()

#scaffold = tf.train.Scaffold(summary_op=merged)
summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                         output_dir="test/logs",
                                         summary_op=merged)

profiler_hook = tf.train.ProfilerHook(save_steps=10,
                                      output_dir="test/logs",
                                      show_memory=True)

#scaffold.finalize()

# Start training
with tf.train.MonitoredSession(hooks=[summary_hook, profiler_hook],
                               session_creator=tf.train.ChiefSessionCreator(
                                   config=tf.ConfigProto(log_device_placement=True,
                                                         graph_options=tf.GraphOptions(build_cost_model=1)))) as sess:
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run([train_op, merged], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8}, run_metadata=run_metadata, options=run_options)

        if step % display_step == 0 or step == 1:
            #Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})

            print("after sess.run")
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))


cost_graph = run_metadata.cost_graph
print("cost graph, number of nodes = ", len(cost_graph.node))
run_metadata_str = run_metadata.SerializeToString()

with open("run_metadata.dat", "wb") as metadata_fobj:
    metadata_fobj.write(run_metadata_str)
