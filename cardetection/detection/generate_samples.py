# import os.path
# import yaml
# import cv2
# import cardetection.carutils.strutils as strutils
# import cardetection.carutils.fileutils as fileutils
import itertools
import random
import numpy as np
import cv2
import cardetection.carutils.images as utils
import cardetection.carutils.geometry as gm
from cardetection.detection.detector import ObjectDetector

# TODO: Move these methods to a more sensible module.
def load_negative_region_generator(config_yaml):
    window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
    bak_img_dir = config_yaml['dataset']['directory']['generation']['input']['background']
    modifiers = config_yaml['dataset']['modifiers']
    return generate_negative_regions(bak_img_dir, window_dims, modifiers)
def load_exclusion_region_generator(config_yaml):
    window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
    bak_img_dir = config_yaml['dataset']['directory']['generation']['input']['exclusion']
    bbinfo_dir = config_yaml['dataset']['directory']['bbinfo']
    exl_info_map = utils.load_opencv_bounding_box_info_directory(bbinfo_dir, suffix='exclusion')
    modifiers = config_yaml['dataset']['modifiers']
    return generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims, modifiers)
def load_hard_negative_region_generator(config_yaml):
    window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
    bak_img_dir = config_yaml['dataset']['directory']['generation']['input']['background']
    classifier_dir = config_yaml['dataset']['directory']['generation']['input']['hard_negative_classifier']
    # modifiers = config_yaml['dataset']['modifiers']
    return generate_hard_negative_regions(bak_img_dir, classifier_dir, window_dims)
def load_positive_region_generator(config_yaml):
    window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
    print 'window_dims', window_dims
    pos_img_dir = config_yaml['dataset']['directory']['generation']['input']['positive']
    bbinfo_dir = config_yaml['dataset']['directory']['bbinfo']
    modifiers = config_yaml['dataset']['modifiers']
    return generate_positive_regions(pos_img_dir, bbinfo_dir, modifiers, window_dims)

def generate_hard_negative_regions(bak_img_dir, classifier_dir, window_dims):
    all_images = utils.list_images_in_directory(bak_img_dir)

    w, h = window_dims
    window_aspect = w / float(h)

    with ObjectDetector(classifier_dir) as detector:
        while True:
            random.shuffle(all_images)

            for img_path in all_images:
                # Detect objects:
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                img_dims = (img_w, img_h)
                opencv_rects, scaled_img_dims = detector.detect_objects_in_image(
                    img,
                    resize=False,
                    return_detection_img=False,
                    progress=False
                )
                pixel_rects = map(gm.PixelRectangle.from_opencv_bbox, opencv_rects)

                # Find rectangles in original image dimensions:
                pixel_rects = [rect.scale_image(scaled_img_dims, img_dims) for rect in pixel_rects]

                # Enlarge to window_dims:
                pixel_rects = [rect.enlarge_to_aspect(window_aspect) for rect in pixel_rects]

                # Ensure new rectangles are located within their images:
                pixel_rects = [rect.translated([0,0], img_dims) for rect in pixel_rects]
                pixel_rects = [rect for rect in pixel_rects if rect.lies_within_frame(img_dims)]

                # Convert to image regions:
                image_regions = [utils.ImageRegion(rect, img_path) for rect in pixel_rects]

                # Yield results:
                for reg in image_regions:
                    yield reg

# generate_positive_regions :: String -> Map String ??? -> String -> (Int, Int) -> generator(ImageRegion)
def generate_positive_regions(image_dir, bbinfo_dir, modifiers_config=None, window_dims=None, min_size=(48,48)):
    print 'generate_positive_regions:'
    all_images = utils.list_images_in_directory(image_dir)

    # Create the modifier generator:
    # Note: Simply generate null modifiers if no modifier config is passed.
    modifier_generator = itertools.repeat(utils.RegionModifiers())
    if modifiers_config:
        modifier_generator = utils.RegionModifiers.random_generator_from_config_dict(modifiers_config)

    # Filter out images without bounding boxes:
    bbinfo_map = utils.load_opencv_bounding_box_info_directory(bbinfo_dir, suffix='bbinfo')
    source_images = [img_path for img_path in all_images if not utils.info_entry_for_image(bbinfo_map, img_path) is None]

    # Extract image regions:
    source_regions = []
    for img_path in source_images:
        rects = utils.info_entry_for_image(bbinfo_map, img_path)
        for rect in rects:
            # Reject small samples:
            if rect.w < float(min_size[0]) or rect.h < float(min_size[1]):
                continue

            region = utils.ImageRegion(rect, img_path)
            source_regions.append(region)

    print 'Found {} source regions.'.format(len(source_regions))

    # Generate an infinite list of samples:
    while True:
        # Randomise the order of source images:
        # Note: Don't randomise if no modifiers are used.
        if modifiers_config:
            random.shuffle(source_regions)

        # For each source region:
        for reg in source_regions:
            mod = modifier_generator.next()

            # Enlarge to correct aspect ratio:
            new_rect = reg.rect
            if window_dims:
                aspect = window_dims[0] / float(window_dims[1])
                new_rect = new_rect.enlarge_to_aspect(aspect)
                imsize = utils.get_image_dimensions(reg.fname)
                if not new_rect.lies_within_frame(imsize):
                    # print 'bad', new_rect
                    continue
                # else:
                #     print new_rect

            # Assign a random modifier, and yield the region:
            new_reg = utils.ImageRegion(new_rect, reg.fname, mod)
            yield new_reg

        # If we're not using modifiers, only use each region once:
        if not modifiers_config:
            break


def generate_negative_regions(image_dir, window_dims, modifiers_config=None):
    print 'generate_negative_regions:'
    image_list = utils.list_images_in_directory(image_dir)
    if len(image_list) == 0:
        raise ValueError('The given directory \'{}\' contains no images.'.format(image_dir))
    print '  Found {} images.'.format(len(image_list))

    # Create the modifier generator:
    # Note: Simply generate null modifiers if no modifier config is passed.
    modifier_generator = itertools.repeat(utils.RegionModifiers())
    if modifiers_config:
        modifier_generator = utils.RegionModifiers.random_generator_from_config_dict(modifiers_config)

    min_w, min_h = window_dims
    min_size_length = min(min_w, min_h) / 2.0

    while True:
        random.shuffle(image_list)

        for img_path in image_list:
            imsize = utils.get_image_dimensions(img_path)

            # Reject small images:
            w, h = imsize
            if w < min_w or h < min_h:
                continue

            # Generate a number of regions per image:
            for i in xrange(10):
                rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize, min_size_length)
                mod = modifier_generator.next()
                reg = utils.ImageRegion(rect, img_path, mod)
                yield reg

def generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims, modifiers_config=None):
    print 'generate_negative_regions_with_exclusions:'
    all_images = utils.list_images_in_directory(bak_img_dir)
    if len(all_images) == 0:
        raise ValueError('The given directory \'{}\' contains no images.'.format(bak_img_dir))
    print '  Found {} images.'.format(len(all_images))

    all_images = [img_path for img_path in all_images if not utils.info_entry_for_image(exl_info_map, img_path) is None]
    print '  Found {} images with exclusion info.'.format(len(all_images))

    if len(all_images) == 0:
        raise ValueError('The given directory \'{}\' contains no images with exclusion info.'.format(bak_img_dir))

    image_list = all_images
    random.shuffle(image_list)

    modifier_generator = itertools.repeat(utils.RegionModifiers())
    if modifiers_config:
        modifier_generator = utils.RegionModifiers.random_generator_from_config_dict(modifiers_config)

    while True:
        for img_path in image_list:
            imsize = utils.get_image_dimensions(img_path)

            for reg in generate_negative_regions_in_image_with_exclusions(img_path, exl_info_map, window_dims):
                mod = modifier_generator.next()

                # Assign a random modifier, and yield the region:
                new_reg = utils.ImageRegion(reg.rect, reg.fname, mod)
                yield new_reg

        # Shuffle the image list:
        random.shuffle(image_list)

def generate_negative_regions_in_image_with_exclusions(img_path, exl_info_map, window_dims):
    excl_info = utils.info_entry_for_image(exl_info_map, img_path)
    # Only consider images with exclusions:
    if excl_info is None:
        return
    imsize = utils.get_image_dimensions(img_path)

    img_w, img_h = imsize

    aspect = window_dims[0] / float(window_dims[1])

    scale_step = 1.25
    window_step = 0.5
    max_scale = 0.5
    min_w = 64
    min_scale = min_w / float(img_w)
    scale = min_scale

    # Used to probabilistically reject a fraction of samples at each scale
    # level. This decreases the difference in the number of samples selected
    # from each scale level (otherwise, small scale levels would contain an
    # overwhelmingly large number of sample windows).
    def accept_sample(curr_scale):
        f = (curr_scale - min_scale) / (max_scale - min_scale)
        min_prob = 0.3**0.5
        max_prob = 1.0
        prob = min_prob*(1-f) + max_prob*(f)
        return np.random.uniform() < prob*prob

    # Probability of rejecting samples that are not close to exclusion regions.
    # (Most images have large portions of sky and ground, and this prevents
    # oversampling those areas)
    far_reject_prob = 0.97

    num_found = 0
    while True:
        w = int(round(img_w * scale))
        h = int(round(img_w * scale / aspect))
        sw = int(window_step*w)
        sh = int(window_step*h)
        # print 'scale, (w, h), (sw, sh):', scale, (w, h), (sw, sh)

        if scale > max_scale:
            return
        scale *= scale_step

        count = 0
        for x in xrange(0, img_w, sw):
            for y in xrange(0, img_h, sh):
                if x + w > img_w or y + h > img_h:
                    break

                # Randomly reject samples based on scale:
                if not accept_sample(scale):
                    continue

                rect = gm.PixelRectangle.from_opencv_bbox([x, y, w, h])

                # Nudge rectangles to avoid exclusion regions:
                intersecting = filter(rect.intersects_pixelrectangle, excl_info)
                if len(intersecting) == 1:
                    excl = intersecting[0]
                    rect, offset = rect.moved_to_clear(excl, return_offset=True)
                    if abs(offset[0]) >= sw or  abs(offset[1]) >= sh:
                        continue
                    rect = rect.translated((0,0), imsize)

                # Ensure the rectangle does not intersect an exclusion region:
                if not any((rect.intersects_pixelrectangle(pr) for pr in excl_info)):
                    # Determing whether the rectangle is close to an exclusion
                    # region:
                    is_close_to_exclusion = any((rect.distance_pixelrectangle(pr) < w for pr in excl_info))

                    # Prefer rectangles that are close to exclusion regions:
                    if not is_close_to_exclusion:
                        if np.random.uniform() < far_reject_prob:
                            continue

                    reg = utils.ImageRegion(rect, img_path)
                    num_found += 1
                    # import sys
                    # sys.stdout.write(str(num_found) + ',')
                    # sys.stdout.flush()
                    yield reg
                    count += 1
        # print count


# def generate_random_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims):
#     print 'generate_negative_regions:'
#     all_images = utils.list_images_in_directory(bak_img_dir)
#     if len(all_images) == 0:
#         raise ValueError('The given directory \'{}\' contains no images.'.format(bak_img_dir))
#     print '  Found {} images.'.format(len(all_images))
#
#     all_images = [img_path for img_path in all_images if not utils.info_entry_for_image(exl_info_map, img_path) is None]
#     print '  Found {} images with exclusion info.'.format(len(all_images))
#
#     if len(all_images) == 0:
#         raise ValueError('The given directory \'{}\' contains no images with exclusion info.'.format(bak_img_dir))
#
#     image_list = all_images
#     random.shuffle(image_list)
#
#     while True:
#         for img_path in image_list:
#             imsize = utils.get_image_dimensions(img_path)
#
#             excl_info = utils.info_entry_for_image(exl_info_map, img_path)
#             # print excl_info
#
#             # Only consider images with exclusions:
#             if excl_info is None:
#                 continue
#
#             # Generate a region that does not intersect an exclusion region:
#             num_attempts = 100
#             for _ in xrange(num_attempts):
#                 # print _
#                 rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
#                 if not any((rect.intersects_pixelrectangle(pr) for pr in excl_info)):
#                     reg = utils.ImageRegion(rect, img_path)
#                     yield reg
#                     break
#
#         # Shuffle the image list:
#         random.shuffle(image_list)
