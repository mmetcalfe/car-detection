import os
import tensorflow as tf
import numpy as np
import input_data

# Variable creation convenience functions:

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

# Convolution and pooling convenience functions:
# Note: Boundaries, stride size, and pooling type can be changed.

def conv2d(x, W, name=None, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)

# From: http://stackoverflow.com/a/33816991/3622526
# See also: https://github.com/tensorflow/tensorflow/issues/908
def conv_mosaic(V, use_col=False):
    """ Reshapes the ouptut of a tf.nn.conv2d layer into a mosaic of images
    representing the output of the convolution. """

    # shape: [col_channels, image_height, image_width, num_filters]
    # i.e. shape: [cc, iy, ix, nf]
    shape = V.get_shape()
    cc = int(shape[0]) if use_col else 1 # col_channels
    iy = int(shape[1]) # image_height
    ix = int(shape[2]) # image_width
    nf = int(shape[3]) # num_filters
    V = tf.slice(V,(0,0,0,0),(cc,-1,-1,-1)) #V[0,...]
    # shape: [cc, iy, ix, nf]

    V = tf.transpose(V,(1, 2, 3, 0))
    # shape: [iy, ix, nf, cc]

    V = tf.reshape(V,(iy,ix,nf*cc))
    # shape: [iy, ix, nf*cc]

    # Padding between images:
    pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
    tx = ix + pad
    ty = iy + pad
    V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
    # shape: [ty, tx, nf*cc]

    # Find appropriate tile-dimensions of the new image:
    cx = int(np.sqrt(nf)) - 1
    while nf%cx != 0:
        cx += 1
    cy = nf // cx
    V = tf.reshape(V,(ty,tx,cy,cx,cc))
    # shape: [ty, tx, cy, cx, cc]

    V = tf.transpose(V,(2,0,3,1,4))
    # shape: [cy, ty, cx, tx, cc]

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*ty,cx*tx,cc))
    # shape: [1, cy*ty, cx*tx, 1]

    return V

def flat_mosaic(V, use_col=False):

    """ Reshapes the ouptut of a readout layer into a mosaic (well, an array) of
    images representing the weights contributing to each class.
    """

    # shape: [num_dense_neurons, num_classes]
    # shape: [flat_img_size, num_class_images]
    shape = V.get_shape()
    flat_img_size = int(shape[0]) # flat_img_size
    nc = int(shape[1]) # num_classes
    # shape: [flat_img_size, nc]

    # Find appropriate (near-square) dimensions for the images:
    ix = int(np.sqrt(flat_img_size)) - 1
    while flat_img_size%ix != 0:
        ix += 1
    iy = flat_img_size // ix

    # print 'V.get_shape()', V.get_shape()
    V = tf.reshape(V, (iy, ix, nc))
    # shape: [iy, ix, nc]

    # Add padding between images:
    pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
    tx = ix + pad
    ty = iy + pad
    V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
    # shape: [ty, tx, nc]

    # Find appropriate tile-dimensions of the new image:
    cx = nc
    cy = 1
    V = tf.reshape(V,(ty,tx,cy,cx))
    # shape: [ty, tx, cy, cx]

    V = tf.transpose(V,(2,0,3,1))
    # shape: [cy, ty, cx, tx]

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*ty,cx*tx,1))
    # shape: [1, cy*ty, cx*tx, 1]

    return V

# def conv_mosaic(V, use_col=False):
#     """ Reshapes the ouptut of a tf.nn.conv2d layer into a mosaic of images
#     representing the output of the convolution. """
#
#     # shape: [col_channels, image_height, image_width, num_filters]
#     # i.e. shape: [cc, iy, ix, nf]
#     shape = V.get_shape()
#     cc = int(shape[0]) if use_col else 1
#     iy = int(shape[1])
#     ix = int(shape[2])
#     num_filters = int(shape[3])
#     V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
#     # shape: [1, iy, ix, nf]
#
#     V = tf.reshape(V,(iy,ix,num_filters))
#     # shape: [iy, ix, nf]
#
#     # Padding between images:
#     pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
#     tx += ix + pad
#     ty += iy + pad
#     V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
#     # shape: [ty, tx, nf]
#
#     cx = 8
#     cy = num_filters // cx
#     V = tf.reshape(V,(ty,tx,cy,cx))
#     # shape: [ty, tx, cy, cx]
#
#     V = tf.transpose(V,(2,0,3,1))
#     # shape: [cy, ty, cx, tx]
#
#     # image_summary needs 4d input
#     V = tf.reshape(V,(1,cy*ty,cx*tx,1))
#     # shape: [1, cy*ty, cx*tx, 1]
#
#     return V

def build_model(x, window_dims, use_colour=True):
    # Placeholder for input images:
    # x = tf.placeholder("float", [None, img_pixels], name='input_images')
    # Build the model:
    # img_w, img_h = 28, 28 # for MNIST
    img_w, img_h = window_dims
    num_col_chnls = 3
    img_pixels = img_w*img_h*num_col_chnls

    x_col_image = tf.reshape(x, [-1, img_h, img_w, num_col_chnls])

    x_image = x_col_image
    if not use_colour:
        num_col_chnls = 1
        x_image = tf.image.rgb_to_grayscale(x_col_image)

    print 'x_col_image.get_shape()', x_col_image.get_shape()
    print 'x_image.get_shape()', x_image.get_shape()

    # First convolutional layer:
    patch_size = 5
    num_in_chnls_1 = num_col_chnls

    # Define weights and bias terms for layer 1:
    # num_out_chnls_1 = 32
    num_out_chnls_1 = 16
    W_conv1 = weight_variable([patch_size, patch_size, num_in_chnls_1, num_out_chnls_1], name='W_conv1')
    print 'W_conv1.get_shape()', W_conv1.get_shape()
    conv = conv2d(x_image, W_conv1)
    print 'conv.get_shape()', conv.get_shape()
    b_conv1 = bias_variable([num_out_chnls_1], name='b_conv1')
    # Convolve x_image with weight tensor, add bias, apply ReLU, then max pool:
    h_conv1 = tf.nn.relu(conv + b_conv1, name='h_conv1')
    h_pool1 = max_pool_2x2(h_conv1, name='h_pool1')

    print 'h_pool1.get_shape()', h_pool1.get_shape()

    # Second convolutional layer:
    # num_out_chnls_2 = 64
    num_out_chnls_2 = 16
    W_conv2 = weight_variable([patch_size, patch_size, num_out_chnls_1, num_out_chnls_2], name='W_conv2')
    b_conv2 = bias_variable([num_out_chnls_2], name='b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
    h_pool2 = max_pool_2x2(h_conv2, name='h_pool2')

    print 'h_pool2.get_shape()', h_pool2.get_shape()

    # Densely connected layer:
    # Note: Image size is now 7x7.
    #       Add a fully connected layer with 1024 neurons to process entire image.
    #       Reshape pooling layer tensor into a batch of vectors, multiply by
    #       weight matrix, add bias, then ReLU.

    h_pool2_shape = h_pool2.get_shape()
    reduced_img_h = int(h_pool2_shape[1])
    reduced_img_w = int(h_pool2_shape[2])
    # num_dense_neurons = 1024
    num_dense_neurons = 256
    reduced_img_pixels = reduced_img_h*reduced_img_w
    W_fc1 = weight_variable([reduced_img_pixels*num_out_chnls_2, num_dense_neurons], name='W_fc1')
    b_fc1 = bias_variable([num_dense_neurons], name='b_fc1')
    print 'W_fc1.get_shape()', W_fc1.get_shape()

    h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_pixels*num_out_chnls_2], name='h_pool2_flat')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

    # Dropout:
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # Readout layer:
    num_classes = 2
    W_fc2 = weight_variable([num_dense_neurons, num_classes], name='W_fc2')
    b_fc2 = bias_variable([num_classes], name='b_fc2')

    print 'W_fc2.get_shape()', W_fc2.get_shape()

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # # Add summary ops to collect data
    # w_hist = tf.histogram_summary("W_conv1", W_conv1, name='w_hist')
    # b_hist = tf.histogram_summary("b_conv1", b_conv1, name='b_hist')
    # y_hist = tf.histogram_summary("logits", logits, name='y_hist')

    # Image summaries:
    tf.image_summary('input', x_image, max_images=10)
    x_image_flat = tf.slice(x_image,(0,0,0,0),(1,-1,-1,-1))
    x_image_flat = tf.reshape(x_image_flat, [-1, img_w*img_h, num_col_chnls], name='x_image_flat')
    x_image_flat = tf.squeeze(x_image_flat) # remove dimensions of size 1
    tf.image_summary('input channels', flat_mosaic(x_image_flat), max_images=1)

    # W_conv1 is (5, 5, in_chnls, out_chnls) == (h, w, in_chnls, out_chnls)
    # Transpose to (in_chnls, h, w, out_chnls)
    W_conv1_transp = tf.transpose(W_conv1,(2,0,1,3), name='W_conv1_transp')
    tf.image_summary('#1 filters W_conv1_transp', conv_mosaic(W_conv1_transp, use_col=True), max_images=1)
    tf.image_summary('#2 filter_results h_conv1', conv_mosaic(h_conv1), max_images=1)

    W_conv2_transp = tf.transpose(W_conv2,(2,0,1,3), name='W_conv2_transp')
    tf.image_summary('#3 filters W_conv2_transp', conv_mosaic(W_conv2_transp), max_images=1)
    tf.image_summary('#4 filter_results h_conv2', conv_mosaic(h_conv2), max_images=1)

    tf.image_summary('#4 filters W_fc2', flat_mosaic(W_fc2), max_images=1)

    regularised_params = [W_fc1, b_fc1, W_fc2, b_fc2]

    return logits, keep_prob, regularised_params

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
    num_data_gen_threads = 7
    datasets = input_data.initialise_data_sets(
        FLAGS.config_yaml,
        pos_frac=0.2,
        # exclusion_frac=0.02,
        exclusion_frac=0.05,
        test_pos_frac=1.0
    )
    feature_batch, label_batch = datasets.train.batch_generators(1000)
    summary_interval = 10
    checkpoint_interval = 50

    # Build the model:
    logits, keep_prob, regularised_params = build_model(
        x=feature_batch,
        window_dims=datasets.train.window_dims
    )

    # Train classifier:
    # Define loss and optimizer:
    # y_ = tf.placeholder("float", [None, num_classes], name="y-input")
    y_ = label_batch

    print 'label_batch.get_shape()', label_batch.get_shape()
    print 'logits.get_shape()', logits.get_shape()
    print 'tf.to_int64(y_).get_shape()', tf.to_int64(y_).get_shape()

    # More name scopes will clean up the graph representation
    with tf.name_scope("loss") as scope:
        # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        # ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.to_int64(y_[:,1])))
        loss_summ = tf.scalar_summary("loss", loss)

        # L2 regularization for the fully connected parameters.
        # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
        #               tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        regularizers = sum([tf.nn.l2_loss(p) for p in regularised_params])
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

    with tf.name_scope("train") as scope:
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimiser = tf.train.AdamOptimizer(1e-4)
        optimiser = tf.train.AdamOptimizer(1e-4, epsilon=1e-6)
        train_step = optimiser.minimize(loss, global_step=global_step)

    # Note: Not using cross_entropy from CIFAR-10 example due to numerical
    # issues.
    # http://stackoverflow.com/a/33713196/3622526
    # http://stackoverflow.com/a/34234606/3622526
    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
    with tf.name_scope("test") as scope:
        y_conv = tf.nn.softmax(logits, name='y_conv')
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

    # Restore the latest checkpoint:
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        # latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        print 'latest_ckpt', latest_ckpt
        print 'get_checkpoint_state', tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, latest_ckpt)

    if True:
        print 'create_threads'
        # enqueue_threads = datasets.train.qrunner.create_threads(sess, coord=datasets.train.coord, start=True)
        datasets.train.start_threads(sess, num_threads=num_data_gen_threads)
        # datasets.train.start_threads(sess, num_threads=1)

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
            i = sess.run(global_step)
            while i < 100000:
                i = sess.run(global_step)
            # for i in range(20000):
                if datasets.train.should_stop():
                    break

                if (i+1)%summary_interval == 0:  # Record summary data, and the accuracy
                    print 'Recording summary data, and accuracy...'
                    # batch_xs, batch_ys = datasets.test.next_batch(100)
                    # # feed = {x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0}
                    # feed = {feature_batch: batch_xs, y_: batch_ys, keep_prob: 1.0}
                    # result = sess.run([merged, accuracy], feed_dict=feed)
                    result = sess.run([merged, accuracy], feed_dict={keep_prob: 1.0})
                    summary_str = result[0]
                    acc = result[1]
                    writer.add_summary(summary_str, i)
                    # print 'Accuracy at step {}: {}.'.format(i, acc)
                    print 'Training accuracy at step {}: {}.'.format(i, acc)

                if (i+1)%checkpoint_interval == 0:
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
