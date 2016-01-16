import glob
import numpy as np
import cv2
import cardetection.carutils.geometry as gm

def resize_sample(sample, shape):
    # Use INTER_AREA for shrinking and INTER_LINEAR for enlarging:
    target_is_smaller = shape[1] < sample.shape[1] # targetWidth < sampleWidth
    interp = cv2.INTER_AREA if target_is_smaller else cv2.INTER_LINEAR
    resized = cv2.resize(sample, (shape[1], shape[0]), interpolation=interp)
    return resized

class ImageRegion(object):
    """ Describes a rectangular region of an image.
    """

    # ImageRegion.__init__ :: PixelRectangle -> String -> ImageRegion
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

    @property
    def as_dict(self):
        info = {}
        info['rect'] = self.rect.opencv_bbox
        info['fname'] = self.fname
        return info

    @classmethod
    def from_dict(cls, info):
        rect = gm.PixelRectangle.from_opencv_bbox(info['rect'])
        fname = info['fname']
        return cls(rect, fname)

class RegionDescriptor(object):
    """ Represents a feature descriptor for a region in an image.
    """

    # RegionDescriptor.__init__ :: ImageRegion -> np.array(dtype=np.float32) -> Int -> RegionDescriptor
    def __init__(self, region, descriptor, label):
        self.region = region
        self.descriptor = descriptor
        self.label = label

    @property
    def as_dict(self):
        info = {}
        info['region'] = self.region.as_dict
        info['descriptor'] = np.squeeze(self.descriptor).tolist()
        info['label'] = self.label
        return info

    @classmethod
    def from_dict(cls, info):
        region = ImageRegion.from_dict(info['region'])
        descriptor = np.array(info['descriptor'], dtype=np.float32)
        label = info['label']
        return cls(region, descriptor, label)

def listImagesInDirectory(image_dir):
    image_list = glob.glob("{}/*.jpg".format(image_dir))
    image_list += glob.glob("{}/*.png".format(image_dir))
    return image_list

def get_hog_info_dict(hog):
    info = {}
    info['winSize'] = hog.winSize
    info['blockSize'] = hog.blockSize
    info['blockStride'] = hog.blockStride
    info['cellSize'] = hog.cellSize
    info['nbins'] = hog.nbins
    info['derivAperture'] = hog.derivAperture
    info['winSigma'] = hog.winSigma
    info['histogramNormType'] = hog.histogramNormType
    info['L2HysThreshold'] = hog.L2HysThreshold
    info['gammaCorrection'] = hog.gammaCorrection
    info['nlevels'] = hog.nlevels
    return info

def create_hog_from_info_dict(info):
    print info
    hog = cv2.HOGDescriptor(
        tuple(info['winSize']),
        tuple(info['blockSize']),
        tuple(info['blockStride']),
        tuple(info['cellSize']),
        info['nbins'],
        info['derivAperture'],
        info['winSigma'],
        info['histogramNormType'],
        info['L2HysThreshold'],
        info['gammaCorrection'],
        info['nlevels'])
    return hog

def hog_info_dicts_match(info_a, info_b):
    result = True
    result &= tuple(info_b['winSize']) == tuple(info_a['winSize'])
    result &= tuple(info_b['blockSize']) == tuple(info_a['blockSize'])
    result &= tuple(info_b['blockStride']) == tuple(info_a['blockStride'])
    result &= tuple(info_b['cellSize']) == tuple(info_a['cellSize'])
    result &= info_b['nbins'] == info_a['nbins']
    result &= info_b['derivAperture'] == info_a['derivAperture']
    result &= info_b['winSigma'] == info_a['winSigma']
    result &= info_b['histogramNormType'] == info_a['histogramNormType']
    result &= info_b['L2HysThreshold'] == info_a['L2HysThreshold']
    result &= info_b['gammaCorrection'] == info_a['gammaCorrection']
    result &= info_b['nlevels'] == info_a['nlevels']
    return result

def name_from_hog_descriptor(hog):
    from strutils import camel_humps_acronym
    from strutils import make_filename_safe

    info = get_hog_info_dict(hog)

    safe_info = {}
    for k, v in info.iteritems():
        safe_name = camel_humps_acronym(k)
        safe_val = make_filename_safe(str(v))
        safe_info[safe_name] = safe_val

    trialName = 'hog_' + '_'.join(map(lambda (k,v): k+v, safe_info.iteritems()))
    return trialName
