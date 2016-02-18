import traceback
import os.path
import itertools
import cv2
import numpy as np
import cardetection.tensorflow.cnnmodel as cnnmodel
import cardetection.carutils.detection as detectutils
import cardetection.carutils.fileutils as fileutils
from progress.bar import Bar as ProgressBar

def draw_detections(img, cars):
    for (x,y,w,h) in cars:
        lw = max(2, img.shape[0] / 100)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),lw)

class TensorFlowObjectDetector(object):
    def __init__(self):
        import tensorflow as tf
        pass
    def cleanup(self):
        self.sess.close()
    @classmethod
    def load_from_directory(cls, checkpoint_dir):
        import tensorflow as tf
        detector = cls()

        config_yaml_fname = fileutils.find_in_ancestors('template.yaml')
        print 'config_yaml_fname:', os.path.abspath(config_yaml_fname)
        config_yaml = fileutils.load_yaml_file(config_yaml_fname)
        detector.window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))

        # Clear all existing TensorFlow variables:
        # Note: This is necessary to prevent errors from the Saver when this
        # method is called more than once.
        # TODO: Find a way to avoid the need to do this.
        tf.reset_default_graph()

        # Placeholder for input images:
        # detector.window_dims = [28, 16]
        print 'detector.window_dims:', detector.window_dims
        img_w, img_h = detector.window_dims
        num_col_chnls = 3
        img_pixels = img_w*img_h*num_col_chnls
        detector.x = tf.placeholder("float", [None, img_pixels], name='input_images')
        y_conv, keep_prob, _ = cnnmodel.build_model(
            x=detector.x,
            window_dims=detector.window_dims
        )
        detector.y_conv = y_conv
        detector.keep_prob = keep_prob

        detector.sess = tf.Session()

        detector.sess.run(tf.initialize_all_variables())

        # Set up the checkpoint saver:
        saver = tf.train.Saver(tf.all_variables())

        # Restore the latest checkpoint:
        rel_checkpoint_dir = os.path.relpath(checkpoint_dir)
        latest_ckpt = tf.train.latest_checkpoint(rel_checkpoint_dir)
        print 'rel_checkpoint_dir', rel_checkpoint_dir
        print 'latest_ckpt', latest_ckpt
        print 'get_checkpoint_state', tf.train.get_checkpoint_state(rel_checkpoint_dir)
        saver.restore(detector.sess, latest_ckpt)

        return detector

    def detect_objects_in_image(self, img, greyscale=False, resize=True, return_detection_img=True, progress=True):
        h, w = img.shape[:2]
        scaled_img_dims = (w, h)
        if resize:
            max_w = 1024
            if img.shape[0] > max_w:
                # print 'resize:', img_path, img.shape
                # img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                h, w = img.shape[:2]
                aspect = w / float(h)
                new_h = int(max_w / aspect)
                img = cv2.resize(img, dsize=(max_w, new_h))
                scaled_img_dims = (max_w, new_h)

        print 'img.shape:', img.shape

        def get_win_gen(only_rects=False):
            return detectutils.sliding_window_generator(img,
                window_dims=self.window_dims,
                scale_factor=1.1,
                strides=(8, 8),
                # scale_factor=1.2,
                # strides=(16, 16),
                only_rects=only_rects
            )

        win_gen = get_win_gen()
        num_windows = sum(1 for _ in get_win_gen(only_rects=True))
        batch_size = 100
        # num_batches = int(np.ceil(num_windows/float(batch_size)))

        detected_cars = []

        progressbar = None
        if progress:
            progressbar = ProgressBar('Processing windows:', max=num_windows, suffix='%(index)d/%(max)d - %(eta)ds')
        while True:
            samples_windows = list(itertools.islice(win_gen, 0, batch_size))
            if len(samples_windows) == 0:
                break

            # Unzip the list of tuples into two lists:
            samples, windows = zip(*samples_windows)

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns*depth]
            samples = np.stack(samples)
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2] * samples.shape[3])

            feed = {self.x: samples, self.keep_prob: 1.0}
            label_probs = self.sess.run(self.y_conv, feed_dict=feed)

            if progress:
                progressbar.next(batch_size)

            for probs, window in zip(label_probs, windows):
                pos_prob, neg_prob = probs
                if pos_prob > neg_prob:
                    detected_cars.append(window.opencv_bbox)

        if progress:
            progressbar.finish()

        if len(detected_cars) > 0:
            detected_cars = np.stack(detected_cars)
        else:
            detected_cars = np.array([])

        if return_detection_img:
            draw_detections(img, detected_cars)
            return detected_cars.tolist(), img
        else:
            return detected_cars.tolist(), scaled_img_dims

class OpenCVObjectDetector(object):
    def __init__(self, classifier):
        self.classifier = classifier
    def cleanup(self):
        pass
    @classmethod
    def load_from_directory(cls, data_dir):
        classifier_xml = '{}/cascade.xml'.format(data_dir)

        if not os.path.isdir(data_dir):
            classifier_xml = data_dir
            if not os.path.isfile(classifier_xml):
                raise ValueError('The given file \'{}\' does not exist.'.format(classifier_xml))
        elif not os.path.isfile(classifier_xml):
            raise ValueError('The directory \'{}\' does not contain a trained cascade classifier (cascade.xml).'.format(data_dir))

        classifier = cv2.CascadeClassifier(classifier_xml)
        return cls(classifier)
    def detect_objects_in_image(self, img, greyscale=True, resize=True, return_detection_img=True, progress=True):
        # if resize:
        #     while img.shape[0] > 1024:
        #         # print 'resize:', img_path, img.shape
        #         img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

        h, w = img.shape[:2]
        scaled_img_dims = (w, h)
        if resize:
            max_w = 1024
            if img.shape[0] > max_w:
                # print 'resize:', img_path, img.shape
                # img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                h, w = img.shape[:2]
                aspect = w / float(h)
                new_h = int(max_w / aspect)
                img = cv2.resize(img, dsize=(max_w, new_h))
                scaled_img_dims = (max_w, new_h)

        # # Check whether the image is upside-down:
        # if checkImageOrientation(img_path):
        #     print 'Flipped!'
        #     img = cv2.flip(img, -1)

        gray = img
        if greyscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        minSize = (img.shape[0] / 50, img.shape[1] / 50)
        detected_cars = self.classifier.detectMultiScale(
            image=gray,
            # scaleFactor=1.05,
            scaleFactor=1.01,
            minNeighbors=4,
            minSize=minSize,
        )

        if len(detected_cars) <= 0:
            detected_cars = np.array([])

        if return_detection_img:
            draw_detections(img, detected_cars)
            return detected_cars.tolist(), img
        else:
            return detected_cars.tolist(), scaled_img_dims


# TODO: Make abstract and provide OpenCV and TensorFlow implementations.
# See: https://docs.python.org/2/library/abc.html
class ObjectDetector(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __enter__(self):
        self.detector = self.load_from_directory(self.data_dir)
        return self.detector

    def __exit__(self, exc_type, exc_value, traceback):
        self.detector.cleanup()

    @classmethod
    def load_from_directory(cls, data_dir):
        detector = None
        try:
            detector = OpenCVObjectDetector.load_from_directory(data_dir)
        except ValueError: # TODO: Use a custom exception here.
            try:
                detector = TensorFlowObjectDetector.load_from_directory(data_dir)
            except ValueError: # TODO: Use a custom exception here.
                # exc_info = sys.exc_info()
                # print exc_info
                traceback.print_exc()
                raise ValueError('The directory \'{}\' does not contain a valid OpenCV or TensorFlow classifier.'.format(data_dir))

        return detector
