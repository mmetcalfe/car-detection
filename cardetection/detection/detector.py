import traceback
import os.path
import itertools
import cv2
import numpy as np
import cardetection.tensorflow.cnnmodel as cnnmodel
import cardetection.carutils.images as utils
import cardetection.carutils.detection as detectutils
import cardetection.carutils.fileutils as fileutils
import cardetection.carutils.geometry as gm
from progress.bar import Bar as ProgressBar
import tensorflow as tf

def draw_detections(img, cars, col=(255,0,0)):
    for (x,y,w,h) in cars:
        lw = max(2, img.shape[0] / 100)
        cv2.rectangle(img,(x,y),(x+w,y+h),col,lw)

class TensorFlowObjectDetector(object):
    def __init__(self):
        pass
    def cleanup(self):
        self.sess.close()
    @classmethod
    def load_from_directory(cls, checkpoint_dir):
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
        detector.x_img = tf.placeholder("float", [img_h, img_w, num_col_chnls], name='x_img')
        detector.x = tf.placeholder("float", [None, img_pixels], name='input_images')
        logits, keep_prob, _ = cnnmodel.build_model(
            x=detector.x,
            window_dims=detector.window_dims
        )
        detector.logits = logits
        detector.keep_prob = keep_prob

        init = tf.initialize_all_variables()
        detector.sess = tf.Session(
            # config=tf.ConfigProto(
            #     inter_op_parallelism_threads=1,
            #     intra_op_parallelism_threads=1
            # )
        )
        # print 'Session created!'
        print detector.sess
        detector.sess.run(init)
        # print 'Variables initialised!'

        # Set up the checkpoint saver:
        saver = tf.train.Saver(tf.all_variables())

        # Restore the latest checkpoint:
        rel_checkpoint_dir = os.path.relpath(checkpoint_dir)
        latest_ckpt = tf.train.latest_checkpoint(rel_checkpoint_dir)
        print 'rel_checkpoint_dir', rel_checkpoint_dir
        print 'latest_ckpt', latest_ckpt
        print 'get_checkpoint_state', tf.train.get_checkpoint_state(rel_checkpoint_dir)
        saver.restore(detector.sess, latest_ckpt)

        # print 'Model restored!'

        return detector

    def prepare_sample(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert from [0, 255] -> [0.0, 1.0].
        scaled_img = np.multiply(rgb_img, 1.0 / 255.0)

        feed = {self.x_img: scaled_img}
        float_img = tf.cast(self.x_img, tf.float32)

        whitening_op = tf.image.per_image_whitening(float_img)
        sample = self.sess.run(whitening_op, feed_dict=feed)
        return sample

    def classify(self, img, pixel_rect):
        img_h, img_w = img.shape[:2]
        img_dims = (img_w, img_h)
        w, h = self.window_dims
        window_shape = (h, w)
        window_aspect = w / float(h)
        # print window_aspect

        # Enlarge to window_dims:
        enlarged_rect = pixel_rect.enlarge_to_aspect(window_aspect)
        # Ensure rectangle is located within its image:
        object_rect = enlarged_rect.translated([0,0], img_dims)
        # if not pixel_rect.lies_within_frame(img_dims):

        sample = utils.crop_rectangle(img, object_rect)
        sample = utils.resize_sample(sample, shape=window_shape)
        sample = self.prepare_sample(sample)

        flat_sample = sample.reshape(sample.shape[0] * sample.shape[1] * sample.shape[2])
        feed = {self.x: [flat_sample], self.keep_prob: 1.0}
        label_probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed)

        pos_prob, neg_prob = label_probs[0]
        is_object = pos_prob > neg_prob

        return is_object, object_rect, pos_prob

    def detect_objects_in_image(self, img, greyscale=False, resize=True, return_detection_img=True, progress=True):
        h, w = img.shape[:2]
        scaled_img_dims = (w, h)
        if resize:
            max_w = 1024
            # max_w = 200
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

            # Perform the required colour conversion and preprocessing:
            samples = [self.prepare_sample(sample) for sample in samples]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns*depth]
            samples = np.stack(samples)
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2] * samples.shape[3])

            feed = {self.x: samples, self.keep_prob: 1.0}
            label_probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed)

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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

class CascadedObjectDetector(object):
    def __init__(self, detectors):
        self.detectors = detectors
    def cleanup(self):
        for detector in self.detectors:
            detector.cleanup()
    @classmethod
    def load_from_directory(cls, data_dir_lst):
        detectors = []
        for data_dir in data_dir_lst:
            detector = ObjectDetector.load_from_directory(data_dir)
            detectors.append(detector)
        return cls(detectors)
    def detect_objects_in_image(self, img, greyscale=True, resize=True, return_detection_img=True, progress=True):
        first_detector, other_detectors = self.detectors[0], self.detectors[1:]

        # Detect objects with first classifier:
        img_h, img_w = img.shape[:2]
        img_dims = (img_w, img_h)
        opencv_rects, scaled_img_dims = first_detector.detect_objects_in_image(
            img,
            greyscale=greyscale,
            resize=resize,
            return_detection_img=False,
            progress=progress
        )
        pixel_rects = map(gm.PixelRectangle.from_opencv_bbox, opencv_rects)

        # Find rectangles in original image dimensions:
        pixel_rects = [rect.scale_image(scaled_img_dims, img_dims) for rect in pixel_rects]

        detector_rects = [pixel_rects]
        for detector in other_detectors:
            object_rects = []
            for rect in detector_rects[-1]:
                # Attempt to classify the object at several different scales:
                num_scales = 5
                max_scale_diff = 0.3
                for scale in np.linspace(1.0 - max_scale_diff, 1.0 + max_scale_diff, num_scales):
                    test_rect = rect.scaled_about_center((scale, scale), img_dims)
                    is_object, object_rect, prob = detector.classify(img, test_rect)
                    # print is_object, object_rect
                    if prob > 0.9:
                    # if True:
                        object_rects.append(object_rect)
                        break
                    elif is_object:
                        print 'WARNING: Weak positive rejected (prob={})'.format(prob)

            detector_rects.append(object_rects)

        # print 'detector_rects:', detector_rects

        detected_cars_lst = [rect.opencv_bbox for rect in detector_rects[-1]]
        if return_detection_img:
            # TODO: Draw output of each classifier.
            cols = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]
            for i, rects in enumerate(detector_rects):
                opencv_rects = [rect.opencv_bbox for rect in rects]
                draw_detections(img, opencv_rects, cols[i])
            return detected_cars_lst, img
            # return detected_cars_lst, reg_img
        else:
            return detected_cars_lst, img_dims

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
        if isinstance(data_dir, (list, tuple)):
            return CascadedObjectDetector.load_from_directory(data_dir)

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
