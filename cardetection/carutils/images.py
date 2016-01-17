import glob
import numpy as np
import cv2
import cardetection.carutils.geometry as gm

def resize_sample(sample, shape, use_interp=True):
    # Use INTER_AREA for shrinking and INTER_LINEAR for enlarging:
    interp = cv2.INTER_NEAREST
    if use_interp:
        target_is_smaller = shape[1] < sample.shape[1] # targetWidth < sampleWidth
        interp = cv2.INTER_AREA if target_is_smaller else cv2.INTER_LINEAR
    resized = cv2.resize(sample, (shape[1], shape[0]), interpolation=interp)
    return resized

class RegionModifiers(object):
    """ Specifies how an image region should be modified to obtain a training sample.
    """

    def __init__(self):
        self.use_interp = True
        self.xflip = False
        self.rotation = 0.0
        self.trans_frac = [0.0, 0.0] # Translation as a fraction of sample dimensions.
        self.scale_factors = [1.0, 1.0]

    def __repr__(self):
        return '{{RegionModifiers | xflip:{}, rotation:{}, trans_frac:{}, scale_factors:{}}}'.format(self.xflip, self.rotation, self.trans_frac, self.scale_factors)

    @property
    def as_dict(self):
        info = {}
        info['use_interp'] = self.use_interp
        info['xflip'] = self.xflip
        info['rotation'] = self.rotation
        info['trans_frac'] = self.trans_frac
        info['scale_factors'] = self.scale_factors
        return info

    @classmethod
    def from_dict(cls, info):
        if not info:
            return None
        if not 'xflip' in info:
            return None

        reg_mod = cls()
        reg_mod.use_interp = (info['use_interp'] == 'True') or (info['use_interp'] == True)
        reg_mod.xflip = (info['xflip'] == 'True') or (info['xflip'] == True)
        reg_mod.rotation = float(info['rotation'])
        reg_mod.trans_frac = map(float, info['trans_frac'])
        reg_mod.scale_factors = map(float, info['scale_factors'])
        return reg_mod

    # Use the modifiers section of the template.yaml file to create a generator
    # that creates an infinite stream of random modifiers.
    @staticmethod
    def random_generator_from_config_dict(modifiers_config):
        use_interp = modifiers_config['use_interp'] == 'True' or (modifiers_config['use_interp'] == True)
        enable_xflip = modifiers_config['enable_xflip'] == 'True' or (modifiers_config['enable_xflip'] == True)
        max_rotation_degrees = float(modifiers_config['max_rotation_degrees'])
        max_trans_frac = map(float, modifiers_config['max_trans_frac'])
        scale_range = map(float, modifiers_config['scale_range'])
        scale_axes_independently = modifiers_config['scale_axes_independently'] == 'True' or (modifiers_config['scale_axes_independently'] == True)

        while True:
            xflip = np.random.uniform() < 0.5
            x_trans = np.random.uniform(-max_trans_frac[0], max_trans_frac[0])
            y_trans = np.random.uniform(-max_trans_frac[1], max_trans_frac[1])
            rotation = np.random.uniform(-max_rotation_degrees, max_rotation_degrees)
            x_scale = np.random.uniform(scale_range[0], scale_range[1])
            y_scale = np.random.uniform(scale_range[0], scale_range[1])

            reg_mod = RegionModifiers()
            reg_mod.use_interp = use_interp
            reg_mod.xflip = xflip
            reg_mod.rotation = rotation
            reg_mod.trans_frac = [x_trans, y_trans]
            if scale_axes_independently:
                reg_mod.scale_factors = [x_scale, y_scale]
            else:
                reg_mod.scale_factors = [x_scale, x_scale]

            yield reg_mod


class ImageRegion(object):
    """ Describes a rectangular region of an image, and how to modify it to obtain a sample.
    """

    # ImageRegion.__init__ :: PixelRectangle -> String -> ImageRegion
    def __init__(self, rect, fname, modifiers = None):
        self.rect = rect
        self.fname = fname
        self.modifiers = modifiers

    def load_cropped_sample(self, modifiers=None):
        if not modifiers:
            modifiers = self.modifiers

        img = cv2.imread(self.fname)
        if not modifiers:
            # If there are no modifiers, simply crop and return the sample:
            cropped = img[self.rect.y1:self.rect.y2, self.rect.x1:self.rect.x2, :]
            return cropped
        else:
            # Apply the modifiers:
            modified = img
            mod_rect = self.rect
            dims = (modified.shape[1], modified.shape[0])

            # Is probably faster to apply this to the cropped sample:
            # # Apply xflip:
            # if modifiers.xflip:
            #     # 1 for y-axis flip (Note: the OpenCV documentation is misleading)
            #     modified = cv2.flip(modified, 1)
            #     mod_rect = mod_rect.mirrored_x(dims)

            # Apply translation:
            trans_vec = np.array(modifiers.trans_frac) * np.array([mod_rect.w, mod_rect.h])
            mod_rect = mod_rect.translated(trans_vec, dims)

            # Apply scaling:
            scale_factors = modifiers.scale_factors
            mod_rect = mod_rect.scaled_about_center(scale_factors, dims)

            # Apply rotation:
            if modifiers.rotation:
                rotation = modifiers.rotation
                M = cv2.getRotationMatrix2D(self.rect.exact_centre, rotation, 1)
                modified = cv2.warpAffine(modified, M, dims)

            # Crop and return the sample:
            cropped = modified[mod_rect.y1:mod_rect.y2, mod_rect.x1:mod_rect.x2, :]

            # Apply xflip:
            if modifiers.xflip:
                cropped = cv2.flip(cropped, 1)

            return cropped

    def load_cropped_resized_sample(self, dimensions, modifiers=None):
        sample = self.load_cropped_sample(modifiers)

        # Determine whether to use interpolation:
        if not modifiers:
            modifiers = self.modifiers
        use_interp = modifiers.use_interp if modifiers else True

        return resize_sample(sample, (dimensions[1], dimensions[0]), use_interp)

    @property
    def as_dict(self):
        info = {}
        info['rect'] = self.rect.opencv_bbox
        info['fname'] = self.fname
        if self.modifiers:
            info['modifiers'] = self.modifiers.as_dict
        return info

    @classmethod
    def from_dict(cls, info):
        rect = gm.PixelRectangle.from_opencv_bbox(info['rect'])
        fname = info['fname']
        modifiers = RegionModifiers.from_dict(info['modifiers'])
        return cls(rect, fname, modifiers)

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
    from strutils import safe_name_from_info_dict
    info = get_hog_info_dict(hog)
    return safe_name_from_info_dict(info, 'hog_')
