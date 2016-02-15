import os
import tensorflow as tf
import input_data

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
    """ Reshapes the ouptut of a tf.nn.conv2d layer into a mosaic of images
    representing the output of the convolution. """

    shape = V.get_shape()
    iy = int(shape[1])
    ix = int(shape[2])
    channels = int(shape[3])
    V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
    V = tf.reshape(V,(iy,ix,channels))

    # Padding between images:
    pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
    ix += pad
    iy += pad
    V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

    cx = 8
    cy = channels // cx
    V = tf.reshape(V,(iy,ix,cy,cx))

    V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*iy,cx*ix,1))

    return V

def build_model(x, window_dims):
    # Placeholder for input images:
    # x = tf.placeholder("float", [None, img_pixels], name='input_images')
    # Build the model:
    # img_w, img_h = 28, 28 # for MNIST
    img_w, img_h = window_dims
    num_col_chnls = 3
    img_pixels = img_w*img_h*num_col_chnls

    x_col_image = tf.reshape(x, [-1, img_h, img_w, num_col_chnls])
    x_image = tf.image.rgb_to_grayscale(x_col_image)
    print 'x_col_image.get_shape()', x_col_image.get_shape()
    print 'x_image.get_shape()', x_image.get_shape()

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

    print 'h_pool1.get_shape()', h_pool1.get_shape()

    # Second convolutional layer:
    num_out_chnls_2 = 64
    W_conv2 = weight_variable([patch_size, patch_size, num_out_chnls_1, num_out_chnls_2])
    b_conv2 = bias_variable([num_out_chnls_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    print 'h_pool2.get_shape()', h_pool2.get_shape()

    # Densely connected layer:
    # Note: Image size is now 7x7.
    #       Add a fully connected layer with 1024 neurons to process entire image.
    #       Reshape pooling layer tensor into a batch of vectors, multiply by
    #       weight matrix, add bias, then ReLU.

    h_pool2_shape = h_pool2.get_shape()
    reduced_img_h = int(h_pool2_shape[1])
    reduced_img_w = int(h_pool2_shape[2])
    num_dense_neurons = 1024
    reduced_img_pixels = reduced_img_h*reduced_img_w
    W_fc1 = weight_variable([reduced_img_pixels*num_out_chnls_2, num_dense_neurons])
    b_fc1 = bias_variable([num_dense_neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_pixels*num_out_chnls_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout:
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer:
    num_classes = 2
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Add summary ops to collect data
    w_hist = tf.histogram_summary("W_conv1", W_conv1)
    b_hist = tf.histogram_summary("b_conv1", b_conv1)
    y_hist = tf.histogram_summary("y_conv", y_conv)

    # Image summaries:
    tf.image_summary('input', x_image, max_images=3)

    # W_conv1 is (5, 5, 1, 32) == (h, w, 1, channels)
    # Transpose to (1, h, w, channels)
    W_conv1_transp = tf.transpose(W_conv1,(2,0,1,3))
    tf.image_summary('filters W_conv1_transp', conv_mosaic(W_conv1_transp), max_images=3)
    tf.image_summary('filter_results h_conv1', conv_mosaic(h_conv1), max_images=3)

    W_conv2_transp = tf.transpose(W_conv2,(2,0,1,3))
    tf.image_summary('filters W_conv2_transp', conv_mosaic(W_conv2_transp), max_images=3)
    tf.image_summary('filter_results h_conv2', conv_mosaic(h_conv2), max_images=3)

    return y_conv, keep_prob

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('train_dir', 'output/cnn-train',
                               """Directory where to write event logs """
                               """and checkpoint.""")

    tf.app.flags.DEFINE_string('config_yaml', 'template.yaml',
                               """Filename of the YAML file describing the """
                               """classifier to train.""")

    # Create the output directory:
    if not os.path.isdir(FLAGS.train_dir):
        raise ValueError('Output directory \'{}\' does not exist!'.format(FLAGS.train_dir))

    # Set up the data source:
    datasets = input_data.initialise_data_sets(
        FLAGS.config_yaml,
        pos_frac=0.2,
        test_pos_frac=1.0
    )
    feature_batch, label_batch = datasets.train.batch_generators(50)

    # Build the model:
    y_conv, keep_prob = build_model(
        x=feature_batch,
        window_dims=datasets.train.window_dims
    )

    # Train classifier:
    # Define loss and optimizer:
    # y_ = tf.placeholder("float", [None, num_classes], name="y-input")
    y_ = label_batch

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
    checkpoint_dir = FLAGS.train_dir
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
        print 'create_threads'
        # enqueue_threads = datasets.train.qrunner.create_threads(sess, coord=datasets.train.coord, start=True)
        datasets.train.start_threads(sess, num_threads=7)

        # One process, 100 steps, batch size 50:
        # real	5m29.924s
        # user	12m43.051s
        # sys	0m44.085s

        # 3 processes, 100 steps, batch size 50:
        # real	3m13.517s
        # user	15m46.751s
        # sys	0m59.183s

        # 7 processes, 100 steps, batch size 50, load batch size: 100
        # real	2m55.385s
        # user	20m43.254s
        # sys	1m10.583s

        # 15 processes, 100 steps, batch size 50:
        # real	3m11.660s
        # user	23m34.095s
        # sys	1m5.392s

        # Train the classifier:
        try:
            for i in range(100):
            # for i in range(20000):
                if datasets.train.should_stop():
                    break

                if (i+1)%10 == 0:  # Record summary data, and the accuracy
                    print 'Recording summary data, and accuracy...'
                    # batch_xs, batch_ys = datasets.test.next_batch(50)
                    # # feed = {x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0}
                    # feed = {x: batch_xs, y_: batch_ys, keep_prob: 1.0}
                    # result = sess.run([merged, accuracy], feed_dict=feed)
                    result = sess.run([merged, accuracy], feed_dict={keep_prob: 1.0})
                    summary_str = result[0]
                    acc = result[1]
                    writer.add_summary(summary_str, i)
                    # print 'Accuracy at step {}: {}.'.format(i, acc)
                    print 'Training accuracy at step {}: {}.'.format(i, acc)

                if (i+1)%100 == 0:
                    print 'Saving checkpoint...'
                    # train_accuracy = accuracy.eval(session=sess,feed_dict={
                    # x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                    saver.save(sess, checkpoint_prefix, global_step=i)
                    print 'Checkpoint saved, step: {}.'.format(i)

                # train_step.run(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
                # print 'train_step'
                print 'train_step {:4}, mpq size: {:4}, tfq size: {:4}'.format(i, datasets.train.get_mp_queue_size(), datasets.train.get_tf_queue_size(sess))
                train_step.run(session=sess, feed_dict={keep_prob: 0.5})
        except Exception, e:
            # Report exceptions to the coordinator.
            datasets.train.request_stop(e)
            print 'Exception:', e
        #    coord.join(threads, stop_grace_period_secs=2)
        datasets.train.stop_threads()
        print 'sess.close()'
        try:
            sess.close()
        except tf.errors.CancelledError:
            pass
    # feed = {x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0}
    # acc = accuracy.eval(session=sess,feed_dict=feed)
    # print 'test accuracy {}'.format(acc)
