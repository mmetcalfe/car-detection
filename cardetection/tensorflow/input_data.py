"""Functions for downloading and reading MNIST data."""
import sys
import Queue
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import itertools
import random
import cardetection.detection.generate_samples as generate_samples
import cardetection.carutils.fileutils as fileutils
from progress.bar import Bar as ProgressBar

# From: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
class SharedCounter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()
    def increment(self, amount=1):
        with self.lock:
            self.val.value += amount
    def decrement(self, amount=1):
        with self.lock:
            self.val.value -= amount
    def value(self):
        with self.lock:
            return self.val.value

def batch_shuffle(gen, batch_size=10000):
    while True:
        batch = list(itertools.islice(gen, 0, batch_size))
        random.shuffle(batch)
        for reg in batch:
            yield reg

class DataGenerator(object):
    def __init__(self, config_yaml_fname, pos_frac, exclusion_frac, classifier_frac):
        config_yaml = fileutils.load_yaml_file(config_yaml_fname)
        self.pos_reg_gen = generate_samples.load_positive_region_generator(config_yaml)
        self.neg_reg_gen = batch_shuffle(generate_samples.load_negative_region_generator(config_yaml), batch_size=100)
        self.exc_reg_gen = batch_shuffle(generate_samples.load_exclusion_region_generator(config_yaml), batch_size=5000)
        self.cls_reg_gen = generate_samples.load_classifier_region_generator(config_yaml)
        self.window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
        self.pos_frac = pos_frac
        self.exc_frac = exclusion_frac
        self.cls_frac = classifier_frac

    # Loads batches in a format compatible with the tensorflow MNIST example.
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        pos_num = int(batch_size*self.pos_frac)
        total_neg_num = batch_size - pos_num
        exc_num = int(total_neg_num*self.exc_frac)
        cls_num = int(total_neg_num*self.cls_frac)
        neg_num = total_neg_num - exc_num - cls_num

        pos_regions = list(itertools.islice(self.pos_reg_gen, 0, pos_num))
        neg_regions = list(itertools.islice(self.neg_reg_gen, 0, neg_num))
        exc_regions = list(itertools.islice(self.exc_reg_gen, 0, exc_num))
        cls_regions = list(itertools.islice(self.cls_reg_gen, 0, cls_num))
        regions = pos_regions + neg_regions + exc_regions + cls_regions

        # Create a tensor containing all images:
        w, h = self.window_dims
        depth = 3
        images = np.zeros((batch_size, h, w, depth), dtype=np.float32)
        # progressbar = ProgressBar('Loading regions', max=batch_size)
        # print 'Loading regions'
        for i, reg in enumerate(regions):
            # progressbar.next()
            sample = reg.load_cropped_resized_sample(self.window_dims)
            images[i,:,:,:] = sample
        # progressbar.finish()

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth]
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])

        # Convert from [0, 255] -> [0.0, 1.0].
        images = np.multiply(images, 1.0 / 255.0)

        # Create one-hot labels:
        pos_labels = np.zeros((pos_num, 2), dtype=np.float32)
        pos_labels[:,0] = 1
        neg_labels = np.zeros((total_neg_num, 2), dtype=np.float32)
        neg_labels[:,1] = 1
        labels = np.row_stack([pos_labels, neg_labels])

        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape, labels.shape)
        )

        # Shuffle the data. This is done last to improve cache performance in
        # load_cropped_resized_sample.
        perm = np.arange(batch_size)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]

        return images, labels

    def sample_generator(self, load_batch_size=100):
        while True:
            batch_images, batch_labels = self.next_batch(load_batch_size)

            for i in xrange(load_batch_size):
                image = batch_images[i]
                label = batch_labels[i]
                yield image, label

    def tf_enqueue_thread(self, dataset, sess, load_batch_size=100):
        # print 'tf_enqueue_thread: sample_generator'
        sample_gen = self.sample_generator(load_batch_size)

        while not dataset.coord.should_stop():
            # print 'tf_enqueue_thread: sample_gen.next()'
            image, label = sample_gen.next()
            feed = {dataset.feature_input: image, dataset.label_input: label}
            dataset.tf_enqueue_op.run(session=sess, feed_dict=feed)

    @staticmethod
    def mp_enqueue_process(mp_queue, qsize_counter, config_yaml_fname, pos_frac, exclusion_frac, load_batch_size=100):
        print 'mp_enqueue_process: DataGenerator'
        sys.stdout.flush()
        data_gen = DataGenerator(config_yaml_fname, pos_frac, exclusion_frac)
        print 'mp_enqueue_process: sample_generator'
        sys.stdout.flush()
        # sample_gen = data_gen.sample_generator(load_batch_size)

        while True:
            # print 'mp_enqueue_process: sample_gen.next()'
            # image, label = sample_gen.next()
            image, label = data_gen.next_batch(load_batch_size)
            feed = (image, label)
            # print 'mp_enqueue_process: mp_queue.put(feed, block=True)'
            # sys.stdout.flush()
            mp_queue.put(feed, block=True)
            qsize_counter.increment()

class DataSet(object):
    # Based on http://stackoverflow.com/a/34596212/3622526
    # https://www.tensorflow.org/versions/0.6.0/how_tos/threading_and_queues/index.html
    def __init__(self, config_yaml_fname, pos_frac, exclusion_frac, maxqsize=1000):
        self.config_yaml_fname = config_yaml_fname
        self.pos_frac = pos_frac
        self.exclusion_frac = exclusion_frac
        self.maxqsize = maxqsize

        config_yaml = fileutils.load_yaml_file(config_yaml_fname)
        self.window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))

        # Create the input data queue:
        w, h = self.window_dims
        depth = 3
        num_classes = 2
        feature_shape = [w*h*depth]
        label_shape = [num_classes]
        self.fifoq = tf.FIFOQueue(
            capacity=maxqsize,
            dtypes=[tf.float32, tf.float32],
            shapes=[feature_shape, label_shape]
        )

        # self.tf_enqueue_op = self.fifoq.enqueue([self.feature_input, self.label_input])
        # self.feature_image = tf.placeholder(tf.float32, shape=[img_h, img_w, depth])
        self.feature_input = tf.placeholder(tf.float32, shape=[None, feature_shape[0]])
        self.label_input = tf.placeholder(tf.float32, shape=[None, label_shape[0]])
        self.tf_enqueue_op = self.fifoq.enqueue_many([self.feature_input, self.label_input])
        self.tf_qsize_op = self.fifoq.size()

        # Use threading or multiprocessing.
        # Threading was found to slow down execution.
        # self.coord = tf.train.Coordinator()
        # self.enqueue_threads = None

        # Process pool and results queue for multiprocessing:
        # self.pool = None
        self.processes = []
        self.mp_qsize = SharedCounter()
        self.mp_queue = multiprocessing.Queue(maxsize=maxqsize)
        self.mp_tf_thread = None
        self.mp_tf_shutdown = False

        # Note: Not using a QueueRunner, as its enqueue operations must be made
        # only of tensorflow graph operations.
        # self.qrunner = tf.train.QueueRunner(self.fifoq, enqueue_ops)

    def get_mp_queue_size(self):
        return self.mp_qsize.value()
    def get_tf_queue_size(self, sess):
        return sess.run(self.tf_qsize_op)

    def transfer_thread(self, sess):
        """ Continuously transfers data from the mp_queue to the tf_queue.
        """
        print 'transfer_thread: start'
        while not self.mp_tf_shutdown:
            # if self.fifoq.size() >= self.maxqsize - 2:
            #     continue

            # print 'transfer_thread: self.mp_queue.get'
            try:
                # image, label = self.mp_queue.get_nowait()
                image, label = self.mp_queue.get(timeout=1)
                self.mp_qsize.decrement()
                # print 'transfer_thread: dataset.tf_enqueue_op.run', 'mpq:', self.mp_qsize.value(), 'tfq:', self.get_tf_queue_size(sess)
                sys.stdout.flush()
                feed = {self.feature_input: image, self.label_input: label}
                self.tf_enqueue_op.run(session=sess, feed_dict=feed)
                # print 'transfer_thread: finish enqueue'
                # sys.stdout.flush()
            except Queue.Empty:
                pass
        print 'transfer_thread: exit'

    def should_stop(self):
        # # Threading implementation:
        # return self.coord.should_stop()
        # Multiprocess implementation:
        return False

    def request_stop(self, exception=None):
        print 'request_stop'
        # # Threading implementation:
        # datasets.train.coord.request_stop(exception)
        # Multiprocess implementation:
        pass

    def start_threads(self, sess, num_threads=4):
        print 'start_threads'
        # # Threading implementation:
        # self.num_threads = num_threads
        # self.data_generators = [DataGenerator(self.config_yaml_fname, self.pos_frac) for i in xrange(self.num_threads)]
        # args = (self,sess)
        # self.enqueue_threads = [threading.Thread(target=dg.enqueue_thread, args=args) for dg in self.data_generators]
        # for th in self.enqueue_threads:
        #     th.start()

        # Multiprocess implementation:
        # Start the generator processes:
        # self.pool = multiprocessing.Pool(processes=num_threads)
        self.processes = []
        args = (self.mp_queue, self.mp_qsize, self.config_yaml_fname, self.pos_frac, self.exclusion_frac)
        for i in xrange(num_threads):
            # print 'self.pool.apply_async'
            # self.pool.apply_async(func=DataGenerator.mp_enqueue_process, args=args)
            p = multiprocessing.Process(target=DataGenerator.mp_enqueue_process, args=args)
            p.start()
            self.processes.append(p)

        # Start the transfer thread:
        args = (self, sess)
        self.mp_tf_thread = threading.Thread(target=DataSet.transfer_thread, args=args)
        print 'self.mp_tf_thread.start()'
        self.mp_tf_thread.start()

    def stop_threads(self):
        # # Threading implementation:
        # print 'stop_threads'
        # # When done, ask the threads to stop.
        # self.coord.request_stop()
        # # And wait for them to actually do it.
        # self.coord.join(self.enqueue_threads)

        # Multiprocess implementation:
        print 'stop_threads'
        self.mp_tf_shutdown = True
        self.fifoq.close(cancel_pending_enqueues=True)
        # self.pool.terminate()
        for p in self.processes:
            p.terminate()

    # def run_loop(self):
    #     # # Launch the graph.
    #     # sess = tf.Session()
    #     # Create a coordinator, launch the queue runner threads.
    #     self.enqueue_threads = self.qrunner.create_threads(sess, coord=coord, start=True)
    #     # Run the training loop, controlling termination with the coordinator.
    #     for step in xrange(1000000):
    #         if coord.should_stop():
    #             break
    #         sess.run(train_op)
    #
    #     self.stop_threads()

    def batch_generators(self, batch_size):
        print 'dequeue_many'
        feature_batch, label_batch = self.fifoq.dequeue_many(batch_size)

        # # Apply per_image_whitening:
        # img_w, img_h = self.window_dims
        # num_col_chnls = 3
        # img_pixels = img_w*img_h*num_col_chnls
        # feature_shape = [img_pixels]
        # whitening_queue = tf.FIFOQueue(
        #     capacity=batch_size,
        #     dtypes=[tf.float32],
        #     shapes=[feature_shape]
        # )
        # whitened_queue = tf.FIFOQueue(
        #     capacity=batch_size,
        #     dtypes=[tf.float32],
        #     shapes=[feature_shape]
        # )
        # whitening_enqueue = whitening_queue.enqueue_many(feature_batch)
        # x_col_image = tf.reshape(whitening_queue.dequeue(), [img_h, img_w, num_col_chnls])
        # whitened = tf.image.per_image_whitening(x_col_image)
        # whitened_flat = tf.reshape(whitened, [img_pixels])
        # whitened_enqueue = whitened_queue.enqueue(whitened_flat)
        # whitened_batch = whitened_queue.dequeue_many(batch_size)

        print feature_batch, label_batch
        # Note: These are tensorflow objects that will automatically refill after
        # each use.
        return feature_batch, label_batch

def initialise_data_sets(config_yaml_fname, pos_frac=0.5, exclusion_frac=0.1, test_pos_frac=0.5):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(config_yaml_fname, pos_frac, exclusion_frac)
    data_sets.test = DataGenerator(config_yaml_fname, test_pos_frac, exclusion_frac)
    # data_sets.validation = DataSet([], 0)
    # data_sets.test = DataSet([], 0)
    return data_sets
