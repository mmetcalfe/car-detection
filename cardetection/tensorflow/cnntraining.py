import os
import tensorflow as tf
import numpy as np
import input_data
import cnnmodel
import time

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
        pos_frac=0.8,
        # exclusion_frac=0.02,
        hard_neg_frac=0.05,
        exclusion_frac=0.05,
        test_pos_frac=1.0
    )
    BATCH_SIZE = 50
    feature_batch, label_batch = datasets.train.batch_generators(BATCH_SIZE)
    SUMMARY_INTERVAL = 1
    CHECKPOINT_INTERVAL = 50

    # Build the model:
    logits, keep_prob, regularised_params = cnnmodel.build_model(
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
            # for i in range(20000):
                if datasets.train.should_stop():
                    break

                # train_step.run(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
                # train_step.run(session=sess, feed_dict={keep_prob: 0.5})
                # print 'train_step'
                start_time = time.time()
                _, loss_value = sess.run([train_step, loss], feed_dict={keep_prob: 0.5})
                step_duration = time.time() - start_time

                examples_per_sec = BATCH_SIZE / float(step_duration)
                sec_per_batch = float(step_duration)
                time_stats = '{:5.1f} examples/sec; {:5.1f} sec/batch'.format(examples_per_sec, sec_per_batch)

                mpq_size = datasets.train.get_mp_queue_size()
                tfq_size = datasets.train.get_tf_queue_size(sess)
                queue_stats = 'mpq size: {:4}, tfq size: {:4}'.format(mpq_size, tfq_size)

                print 'train_step {:4}, {}, loss: {:6.2f}, ({})'.format(i, queue_stats, loss_value, time_stats)

                i = sess.run(global_step)

                if i % SUMMARY_INTERVAL == 0:  # Record summary data, and the accuracy
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

                if i % CHECKPOINT_INTERVAL == 0:
                    print 'Saving checkpoint...'
                    # train_accuracy = accuracy.eval(session=sess,feed_dict={
                    # x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                    saver.save(sess, checkpoint_prefix, global_step=i)
                    print 'Checkpoint saved, step: {}.'.format(i)

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
