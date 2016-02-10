import input_data
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'mnist-train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Variable creation convenience functions:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and pooling convenience functions:
# Note: Boundaries, stride size, and pooling type can be changed.

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# From: http://stackoverflow.com/a/33816991/3622526
# See also: https://github.com/tensorflow/tensorflow/issues/908
def conv_mosaic(V):
    shape = V.get_shape()
    ix = int(shape[2])
    iy = int(shape[1])
    channels = int(shape[3])
    V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
    V = tf.reshape(V,(iy,ix,channels))

    # Padding between images:
    ix += 4
    iy += 4
    V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

    cx = 8
    cy = channels // cx
    V = tf.reshape(V,(iy,ix,cy,cx))

    V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*iy,cx*ix,1))

    return V

# Build the model:
img_w = 28
img_pixels = img_w*img_w

# Placeholder for input images:
x = tf.placeholder("float", [None, img_pixels])

num_col_chnls = 1
x_image = tf.reshape(x, [-1, img_w, img_w, num_col_chnls])

# First convolutional layer:
patch_size = 5
num_in_chnls_1 = 1

# Define weights and bias terms for layer 1:
num_out_chnls_1 = 32
W_conv1 = weight_variable([patch_size, patch_size, num_in_chnls_1, num_out_chnls_1])

b_conv1 = bias_variable([num_out_chnls_1])
# Convolve x_image with weight tensor, add bias, apply ReLU, then max pool:
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer:
num_out_chnls_2 = 64
W_conv2 = weight_variable([patch_size, patch_size, num_out_chnls_1, num_out_chnls_2])
b_conv2 = bias_variable([num_out_chnls_2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer:
# Note: Image size is now 7x7.
#       Add a fully connected layer with 1024 neurons to process entire image.
#       Reshape pooling layer tensor into a batch of vectors, multiply by
#       weight matrix, add bias, then ReLU.

reduced_img_w = 7
num_dense_neurons = 1024
reduced_img_pixels = reduced_img_w*reduced_img_w
W_fc1 = weight_variable([reduced_img_pixels*num_out_chnls_2, num_dense_neurons])
b_fc1 = bias_variable([num_dense_neurons])

h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_pixels*num_out_chnls_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout:
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer:
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Add summary ops to collect data
w_hist = tf.histogram_summary("W_conv1", W_conv1)
b_hist = tf.histogram_summary("b_conv1", b_conv1)
y_hist = tf.histogram_summary("y_conv", y_conv)

# Image summaries:
tf.image_summary('input', x_image, max_images=3)

# W_conv1 is (5, 5, 1, 32) == (h, w, 1, channels)
# Transpose to (1, h, w, channels)
W_conv_transp = tf.transpose(W_conv1,(2,0,1,3))
tf.image_summary('filters', conv_mosaic(W_conv_transp), max_images=3)
tf.image_summary('filter_results', conv_mosaic(h_conv1), max_images=3)

# Train classifier:
# Define loss and optimizer:
y_ = tf.placeholder("float", [None, 10], name="y-input")

# More name scopes will clean up the graph representation
with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)

with tf.name_scope("train") as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# NUM_CORES = 8  # Choose how many cores to use.
# sess = tf.Session(
#     tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
#                    intra_op_parallelism_threads=NUM_CORES))
sess = tf.Session()

# Merge all the summaries and write them out to FLAGS.train_dir:
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)
# tf.initialize_all_variables().run()
sess.run(tf.initialize_all_variables())

# Set up the checkpoint saver:
saver = tf.train.Saver(tf.all_variables())
checkpoint_dir = os.path.join(FLAGS.train_dir)
checkpoint_prefix = os.path.join(FLAGS.train_dir, 'model.ckpt')
print 'FLAGS.train_dir', FLAGS.train_dir
print 'checkpoint_dir', checkpoint_dir
print 'checkpoint_prefix', checkpoint_prefix

if False:
    # Restore the latest checkpoint:
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    print 'latest_ckpt', latest_ckpt
    print 'get_checkpoint_state', tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, latest_ckpt)
else:
    # Train the classifier:
    # for i in range(20000):
    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(50)

        if i % 10 == 0:  # Record summary data, and the accuracy
            feed = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            result = sess.run([merged, accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
            print("Accuracy at step %s: %s" % (i, acc))

        if (i+1)%100 == 0:
            train_accuracy = accuracy.eval(session=sess,feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
            saver.save(sess, checkpoint_prefix, global_step=i)

        train_step.run(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

feed = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
acc = accuracy.eval(session=sess,feed_dict=feed)
print 'test accuracy {}'.format(acc)
