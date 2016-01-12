import glob
import cv2
import cardetection.carutils.geometry as gm

def resize_sample(sample, shape):
    # Use INTER_AREA for shrinking and INTER_LINEAR for enlarging:
    target_is_smaller = shape[1] < sample.shape[1] # targetWidth < sampleWidth
    interp = cv2.INTER_AREA if target_is_smaller else cv2.INTER_LINEAR
    resized = cv2.resize(sample, (shape[1], shape[0]), interpolation=interp)
    return resized

class ImageRegion(object):
    def __init__(self, rect, fname):
        self.rect = rect
        self.fname = fname

    def load_cropped_sample(self):
        img = cv2.imread(self.fname)
        cropped = img[self.rect.y1:self.rect.y2, self.rect.x1:self.rect.x2, :]
        return cropped

    def load_cropped_resized_sample(self, dimensions):
        sample = self.load_cropped_sample()
        return resize_sample(sample, (dimensions[1], dimensions[0]))

def listImagesInDirectory(image_dir):
    image_list = glob.glob("{}/*.jpg".format(image_dir))
    image_list += glob.glob("{}/*.png".format(image_dir))
    return image_list
