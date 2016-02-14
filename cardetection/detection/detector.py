import os.path
import cv2

# TODO: Make abstract and provide OpenCV and TensorFlow implementations.
# See: https://docs.python.org/2/library/abc.html
class ObjectDetector(object):

    def __init__(self, classifier):
        self.classifier = classifier

    @classmethod
    def load_from_directory(cls, data_dir):
        classifier_xml = '{}/cascade.xml'.format(data_dir)

        if not os.path.isfile(classifier_xml):
            raise ValueError('The directory \'{}\' does not contain a trained cascade classifier (cascade.xml).'.format(data_dir))

        classifier = cv2.CascadeClassifier(classifier_xml)
        return cls(classifier)

    def detect_objects_in_image(self, img, greyscale=True, resize=True):
        if resize:
            while img.shape[0] > 1024:
                # print 'resize:', img_path, img.shape
                img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

        # # Check whether the image is upside-down:
        # if checkImageOrientation(img_path):
        #     print 'Flipped!'
        #     img = cv2.flip(img, -1)

        gray = img
        if greyscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        minSize = (img.shape[0] / 50, img.shape[1] / 50)
        cars = self.classifier.detectMultiScale(
            image=gray,
            # scaleFactor=1.05,
            scaleFactor=1.01,
            minNeighbors=4,
            minSize=minSize,
        )

        # print img_path, len(cars)
        if len(cars) > 0:
            for (x,y,w,h) in cars:
                lw = max(2, img.shape[0] / 100)
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),lw)

        return cars.tolist(), img
