import os.path
import random
import argparse
import yaml
import itertools
import numpy as np
import cv2
import cascadetraining as training
import cardetection.carutils.images as utils
import cardetection.carutils.strutils as strutils
import cardetection.carutils.geometry as gm
from progress.bar import Bar as ProgressBar
from cardetection.carutils.datastore import DataStore

def get_svm_detector(svm_file_path):
    # # TODO: Use the SVM_ml.load method these OpenCV issues are fixed:
    # # https://github.com/Itseez/opencv/issues/4969
    # # https://github.com/Itseez/opencv/issues/5891
    # svm = cv2.ml.SVM_create()
    # svm.load(svm_file_path)
    # support_vecs = svm.getSupportVectors()
    # rho, alpha, svidx = svm.getDecisionFunction(0)
    #
    # BUT UNTIL THEN:
    # Manually load the relevant data from the SVM YAML file:
    svm_yaml = None
    # Load yaml file:
    if not os.path.isfile(svm_file_path):
        raise ValueError('Input file \'{}\' does not exist!'.format(svm_file_path))
    with open(svm_file_path, 'r') as fh:
        lines = fh.readlines()

        # Remove invalid YAML version line:
        lines.pop(0)

        # Add spaces after colons to prevent: http://pyyaml.org/wiki/YAMLColonInFlowContext
        # OR remove the bad line:
        lines.pop(7)

        # Remove the (enormous) uncompressed_support_vectors section, to make
        # loading faster:
        usv_begin = None
        usv_end = None
        for i, line in enumerate(lines):
            if 'uncompressed_support_vectors' in line:
                usv_begin = i
            if 'decision_functions' in line:
                usv_end = i
        # print usv_begin, usv_end
        if usv_begin and usv_end:
            del lines[usv_begin:usv_end]

        # Make a string containing the modified file:
        yamlstring = ''.join(lines)

        # Parse the file as YAML:
        import yaml
        # svm_yaml = yaml.load(fh)
        svm_yaml = yaml.load(yamlstring)

    support_vecs = svm_yaml['opencv_ml_svm']['support_vectors']
    rho = svm_yaml['opencv_ml_svm']['decision_functions'][0]['rho']

    # assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    # assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
            #    (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    # assert(sv.type() == CV_32F);

    stacked_vecs = np.hstack(support_vecs)
    svm_detector = np.zeros(len(stacked_vecs)+1, dtype=np.float32)
    svm_detector[:-1] = stacked_vecs
    svm_detector[-1] = float(-rho)

    print 'stacked_vecs', stacked_vecs
    print 'rho', rho
    print 'stacked_vecs.shape', stacked_vecs.shape

    return svm_detector

def get_hog_object(window_dims):
    blockSize = (8,8)
    # blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0 # HOGDescriptor::L2Hys
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(window_dims,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog

def compute_hog_descriptors(hog, image_regions, window_dims, label):
    # hog = get_hog_object(window_dims)

    progressbar = ProgressBar('Computing descriptors', max=len(image_regions))
    reg_descriptors = []
    for reg in image_regions:
        progressbar.next()
        img = reg.load_cropped_resized_sample(window_dims)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        winStride = (8,8)
        padding = (0,0)
        locations = [] # (10, 10)# ((10,20),)
        hist = hog.compute(grey,winStride,padding,locations)

        reg_desc = utils.RegionDescriptor(reg, hist, label)
        reg_descriptors.append(reg_desc)
    progressbar.finish()
    return reg_descriptors

def train_svm(svm_save_path, descriptors, labels):
    # train_data = convert_to_ml(descriptors)
    train_data = np.array(descriptors)
    responses = np.array(labels, dtype=np.int32)

    print "Start training..."
    svm = cv2.ml.SVM_create()
    # Default values to train SVM
    svm.setCoef0(0.0)
    svm.setDegree(3)
    # svm.setTermCriteria(TermCriteria(cv2.TERMCRIT_ITER + cv2.TERMCRIT_EPS, 1000, 1e-3))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3))
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1) # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01) # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR) # C_SVC; # EPSILON_SVR; # may be also NU_SVR; # do regression task
    svm.train(train_data, cv2.ml.ROW_SAMPLE, responses)
    print "...[done]"

    svm.save(svm_save_path)

# def test_classifier(svm_file_path, window_dims):
#     #  Set the trained svm to my_hog
#     hog_detector = get_svm_detector(svm_file_path)
#     hog = get_hog_object(window_dims)
#     hog.setSVMDetector(hog_detector)
#
#     locations = hog.detectMultiScale(img)

def test_classifier(classifier_yaml, svm_file_path, window_dims):
    print 'Loading detector...'
    print svm_file_path
    #  Set the trained svm to my_hog
    hog_detector = get_svm_detector(svm_file_path)
    hog = get_hog_object(window_dims)
    print 'len(hog_detector)', len(hog_detector)
    print 'hog.getDescriptorSize()', hog.getDescriptorSize()
    hog.setSVMDetector(hog_detector)
    print '...[done]'

    for test_dir in classifier_yaml['testing']['directories']:
        test_source_name = test_dir.strip('/').split('/')[-1]

        # results_dir = '{}/{}_results'.format(output_dir, test_source_name)
        # detections_fname = '{}/{}_detections.dat'.format(output_dir, test_source_name)

        img_list = sorted(utils.listImagesInDirectory(test_dir))

        for img_path in img_list:
            img = cv2.imread(img_path)

            h, w = img.shape[:2]
            max_dim = 1200.0
            # if w > max_dim + 50 or h > max_dim + 50:
            if abs(w - max_dim) > 50 or abs(h - max_dim) > 50:
                sx = max_dim / w
                sy = max_dim / h
                current_scale = min(sx, sy)
                img = cv2.resize(img, dsize=None, fx=current_scale, fy=current_scale)

            # img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)

            # while img.shape[1] > 2000:
            #     print 'resize:', img_path, img.shape
            #     img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

            # # Check whether the image is upside-down:
            # if checkImageOrientation(img_path):
            #     print 'Flipped!'
            #     img = cv2.flip(img, -1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            print 'Detecting: {}'.format(img_path)
            # winStride = (4,4)
            winStride = (8,8)
            padding = (0,0)
            # minSize = (img.shape[0] / 50, img.shape[1] / 50)
            cars, weights = hog.detectMultiScale(img,
                winStride=winStride,
                padding=padding,
                scale=1.05,
                useMeanshiftGrouping=False
            )
            print 'cars:', cars
            print 'weights:', weights

            print img_path, len(cars)
            if len(cars) > 0:
                for (x,y,w,h) in cars:
                    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    lw = max(2, min(img.shape[0], img.shape[1]) / 150)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),lw+2)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),lw)
                    # roi_gray = gray[y:y+h, x:x+w]
                    # roi_color = img[y:y+h, x:x+w]

                # img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('img', 500, 500)
                cv2.imshow('img', img)
                # cv2.imshow('img',img)

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: # ESC key
                        cv2.destroyAllWindows()
                        return
                    elif key != 255:
                        break

# choose_positive_samples :: String -> Int -> [String] -> [ImageRegion]
def choose_positive_samples(image_dir, sample_size, bbinfo_dir):
    filtered_image_list = utils.listImagesInDirectory(image_dir)

    # Filter out images without bounding boxes:
    global_info = training.loadGlobalInfo(bbinfo_dir)
    bbox_filter = lambda img_path: os.path.split(img_path)[1] in global_info
    filtered_image_list = filter(bbox_filter, filtered_image_list)

    # Extract image regions:
    img_regions = []
    for img_path in filtered_image_list:
        key = os.path.split(img_path)[1]
        rects = utils.rectangles_from_cache_string(global_info[key])
        for rect in rects:
            region = utils.ImageRegion(rect, img_path)
            img_regions.append(region)

    # Truncate region list to sample the correct number of images:
    if sample_size is None:
        random.shuffle(img_regions)
    else:
        img_regions = random.sample(img_regions, min(sample_size, len(img_regions)))

    if len(img_regions) < sample_size:
        raise ValueError('Requested {} positive samples, but only {} could be found in \'{}\'!'.format(sample_size, len(img_regions), image_dir))

    return img_regions

def view_image_regions(region_generator, dimensions, display_scale):
    for reg in region_generator:
        print 'view_image_regions:'
        print reg.as_dict
        original = reg.load_cropped_resized_sample(dimensions, utils.RegionModifiers())
        sample = reg.load_cropped_resized_sample(dimensions)

        display_shape = (dimensions[1]*display_scale, dimensions[0]*display_scale)
        resized_sample = utils.resize_sample(sample, display_shape, use_interp=False)
        resized_original = utils.resize_sample(original, display_shape, use_interp=False)

        combined_shape = (display_shape[0], display_shape[1]*2)
        combined = np.zeros((combined_shape[0], combined_shape[1], 3), np.uint8)
        cw = display_shape[1]
        combined[:, 0:cw] = resized_original
        combined[:, cw:2*cw] = resized_sample

        cv2.imshow('view_image_regions', combined)
        # cv2.imshow('view_image_regions', resized)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC key
                cv2.destroyAllWindows()
                return
            elif key != 255:
                break

# generate_positive_regions :: String -> Map String ??? -> String -> (Int, Int) -> generator(ImageRegion)
def generate_positive_regions(image_dir, bbinfo_dir, modifiers_config=None, window_dims=None, min_size=(48,48)):
    print 'generate_positive_regions:'
    all_images = utils.listImagesInDirectory(image_dir)

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


def generate_negative_regions(bak_img_dir, neg_num, window_dims):
    print 'generate_negative_regions:'
    all_images = utils.listImagesInDirectory(bak_img_dir)
    if len(all_images) == 0:
        raise ValueError('The given directory \'{}\' contains no images.'.format(bak_img_dir))
    print '  Found {} images.'.format(len(all_images))

    image_list = all_images
    random.shuffle(image_list)

    img_regions = []
    min_w, min_h = window_dims

    progressbar = ProgressBar('Generating negative regions', max=neg_num)
    while True:
        next_img_list = []
        for img_path in image_list:
            imsize = utils.get_image_dimensions(img_path)

            # Reject small regions:
            w, h = imsize
            if w < min_w or h < min_h:
                continue

            # Don't oversample small images:
            # Note: Use an arbitrary, probabilistic rejection strategy.
            prob = (min_w / float(w))
            if np.random.uniform() > prob:
                next_img_list.append(img)

            rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
            reg = utils.ImageRegion(rect, img_path)
            img_regions.append(reg)

            progressbar.next()

            # Exit as soon as we're done:
            if (len(img_regions) >= neg_num):
                # print '  Generated {} regions.'.format(len(img_regions))
                progressbar.finish()
                return img_regions[:neg_num]

        # Swap the image list:
        if len(next_img_list) == 0:
            next_img_list = all_images
        random.shuffle(next_img_list)
        img_list = next_img_list

        # print '  Generated {} regions.'.format(len(img_regions))

def generate_random_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims):
    print 'generate_negative_regions:'
    all_images = utils.listImagesInDirectory(bak_img_dir)
    if len(all_images) == 0:
        raise ValueError('The given directory \'{}\' contains no images.'.format(bak_img_dir))
    print '  Found {} images.'.format(len(all_images))

    all_images = [img_path for img_path in all_images if not utils.info_entry_for_image(exl_info_map, img_path) is None]
    print '  Found {} images with exclusion info.'.format(len(all_images))

    image_list = all_images
    random.shuffle(image_list)

    while True:
        for img_path in image_list:
            imsize = utils.get_image_dimensions(img_path)

            excl_info = utils.info_entry_for_image(exl_info_map, img_path)
            # print excl_info

            # Only consider images with exclusions:
            if excl_info is None:
                continue

            # Generate a region that does not intersect an exclusion region:
            num_attempts = 100
            for _ in xrange(num_attempts):
                # print _
                rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
                if not any((rect.intersects_pixelrectangle(pr) for pr in excl_info)):
                    reg = utils.ImageRegion(rect, img_path)
                    yield reg
                    break

        # Shuffle the image list:
        random.shuffle(image_list)

def generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims, modifiers_config=None):
    print 'generate_negative_regions_with_exclusions:'
    all_images = utils.listImagesInDirectory(bak_img_dir)
    if len(all_images) == 0:
        raise ValueError('The given directory \'{}\' contains no images.'.format(bak_img_dir))
    print '  Found {} images.'.format(len(all_images))

    all_images = [img_path for img_path in all_images if not utils.info_entry_for_image(exl_info_map, img_path) is None]
    print '  Found {} images with exclusion info.'.format(len(all_images))

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

    def accept_sample(curr_scale):
        f = (curr_scale - min_scale) / (max_scale - min_scale)
        min_prob = 0.3**0.5
        max_prob = 1.0
        prob = min_prob*(1-f) + max_prob*(f)
        return np.random.uniform() < prob*prob

    num_found = 0
    while True:
        w = int(round(img_w * scale))
        h = int(round(img_w * scale / aspect))
        sw = int(window_step*w)
        sh = int(window_step*h)
        print scale, (w, h), (sw, sh)

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
                    # Accept only rectangles that are close to exclusion regions:
                    if any((rect.distance_pixelrectangle(pr) < w for pr in excl_info)):
                        reg = utils.ImageRegion(rect, img_path)
                        num_found += 1
                        # import sys
                        # sys.stdout.write(str(num_found) + ',')
                        # sys.stdout.flush()
                        yield reg
                        count += 1
        # print count

def test_random_with_same_aspect():
    # rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
    # if not any((rect.intersects_pixelrectangle(pr) for pr in excl_info)):

    # # Test random_with_same_aspect:
    # img_path = neg_regions[0].fname
    # img = cv2.imread(img_path)
    window_dims = (300, 200)
    img = np.zeros((500, 500, 3), np.uint8)
    h, w = img.shape[:2]
    imsize = (w, h)
    rects = []
    for i in range(10000):
        rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
        rects.append(rect)
        clone = img.copy()
        training.cvDrawRectangle(clone, rect, (255,0,0), 2)

        # Test moved_to_clear:
        new_rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
        training.cvDrawRectangle(clone, new_rect, (0,255,0), 2)
        moved_rect = new_rect.moved_to_clear(rect)
        training.cvDrawRectangle(clone, moved_rect, (0,0,255), 2)
        print rect.intersects_pixelrectangle(moved_rect)

        # Test intersects_pixelrectangle:
        # for i in range(200):
        #     new_rect = gm.PixelRectangle.random_with_same_aspect(window_dims, imsize)
        #     col = (0, 0, 255) if rect.intersects_pixelrectangle(new_rect) else (255, 255, 255)
        #     if not rect.intersects_pixelrectangle(new_rect):
        #         training.cvDrawRectangle(clone, new_rect, col, 2)

        cv2.imshow('img', clone)
        while True:
            key = cv2.waitKey(1) & 0xFF
            # if key == 27: # ESC key
            #     cv2.destroyAllWindows()
            #     return
            if key != 255:
                break

    print 'Saving histogram...'
    from cardetection.carutils.plotting import saveHistogram
    aspects = map(lambda r: r.w, rects)
    saveHistogram('region-aspects.pdf', aspects, bins=20)
    print 'Saved!'


def save_region_descriptors(hog, pos_regions, pos_descriptors, base_fname):
    # Generate an appropriate file name:
    output_dir = 'output'
    descr_name = base_fname
    fname_no_ext = '{}/{}'.format(output_dir,descr_name)
    print 'Saving descriptors to \'{}.xxx\'...'.format(fname_no_ext)

    assert(len(pos_descriptors[0].shape) == 2)
    assert(pos_descriptors[0].shape[1] == 1)

    pairs = zip(pos_regions, pos_descriptors)
    region_descriptors = [{'region': r.as_dict, 'descriptor': np.squeeze(d).tolist()} for r, d in pairs]

    data = {}
    data['hog_descriptor_info'] = utils.get_hog_info_dict(hog)
    data['region_descriptor_list'] = region_descriptors

    # # Save as YAML:
    # with open('{}.yaml'.format(fname_no_ext), 'w') as stream:
    #     yaml.dump(data, stream)

    # Save as pickle (smaller + faster):
    with open('{}.pickle'.format(fname_no_ext), 'wb') as fh:
        import pickle
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)

def load_region_descriptors(trial_file):
    print 'Loading descriptors from \'{}\'...'.format(trial_file)
    fname, ext = os.path.splitext(trial_file)

    data = None
    if ext == '.pickle':
        with open(trial_file, 'rb') as f:
            import pickle
            data = pickle.load(f)
    elif ext == '.yaml':
        data = training.loadYamlFile(trial_file)
    else:
        raise ValueError('The file \'{}\' has an unrecognised extension: \'{}\'.'.format(trial_file, ext))

    hog = utils.create_hog_from_info_dict(data['hog_descriptor_info'])
    regions = []
    descriptors = []
    for entry in data['region_descriptor_list']:
        descriptors.append(np.array(entry['descriptor'], dtype=np.float32))

        rect = gm.PixelRectangle.from_opencv_bbox(entry['region']['rect'])
        fname = entry['region']['fname']
        region = utils.ImageRegion(rect, fname)
        regions.append(region)

    return hog, regions, descriptors

def create_or_load_descriptors(classifier_yaml, hog, window_dims):
    pos_img_dir = classifier_yaml['dataset']['directory']['positive']
    pos_num = int(classifier_yaml['training']['svm']['pos_num'])
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    bak_img_dir = classifier_yaml['dataset']['directory']['background']
    neg_num = int(classifier_yaml['training']['svm']['neg_num'])

    # pos_descriptor_file = 'output/pos_descriptors.pickle'
    # neg_descriptor_file = 'output/neg_descriptors.pickle'
    # if not (os.path.isfile(pos_descriptor_file) and os.path.isfile(neg_descriptor_file)):
    store = DataStore()

    print 'db_name:', store.db_name_for_hog(hog)
    print 'hog_name:', utils.name_from_hog_descriptor(hog)

    # if not store.has_region_descriptors_for_hog(hog):
    # Preprocess samples:
    #   - generate/select sample regions
    #   - compute and descriptors for all samples
    #   - save the regions and descriptors to a file

    curr_pos_num = store.num_region_descriptors(hog, 1)
    req_pos_num = pos_num if curr_pos_num is None else pos_num - curr_pos_num
    print 'pos_num:', pos_num
    print 'curr_pos_num:', curr_pos_num
    print 'req_pos_num:', req_pos_num
    if req_pos_num > 0:
        # if not os.path.isfile(pos_descriptor_file):
        # pos_reg_generator = generate_positive_regions(pos_img_dir, bbinfo_dir, classifier_yaml['dataset']['modifiers'], 0.5*np.array(window_dims))
        pos_reg_generator = generate_positive_regions(pos_img_dir, bbinfo_dir, classifier_yaml['dataset']['modifiers'], window_dims)
        # view_image_regions(pos_reg_generator, window_dims, display_scale=3)
        pos_regions = list(itertools.islice(pos_reg_generator, 0, req_pos_num))
        pos_reg_descriptors = compute_hog_descriptors(hog, pos_regions, window_dims, 1)
        # save_region_descriptors(hog, pos_regions, pos_descriptors, 'pos_descriptors')
        store.save_region_descriptors(pos_reg_descriptors, hog)

    curr_neg_num = store.num_region_descriptors(hog, -1)
    req_neg_num = neg_num if curr_neg_num is None else neg_num - curr_neg_num
    print 'neg_num:', neg_num
    print 'curr_neg_num:', curr_neg_num
    print 'req_neg_num:', req_neg_num
    if req_neg_num > 0:
        # if not os.path.isfile(neg_descriptor_file):
        # neg_regions = generate_negative_regions(bak_img_dir, req_neg_num, window_dims)
        exl_info_map = utils.load_opencv_bounding_box_info_directory(bbinfo_dir, suffix='exclusion')
        neg_reg_generator = generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims, classifier_yaml['dataset']['modifiers'])
        neg_regions = list(itertools.islice(neg_reg_generator, 0, req_neg_num))
        neg_reg_descriptors = compute_hog_descriptors(hog, neg_regions, window_dims, -1)
        # save_region_descriptors(hog, neg_regions, neg_descriptors, 'neg_descriptors')
        store.save_region_descriptors(neg_reg_descriptors, hog)
    # else:
    # Load all descriptors and the hog object used to generate them:
    # hog, reg_descriptors = store.load_region_descriptors_for_hog(hog)
    hog, pos_reg_descriptors = store.load_region_descriptors_for_hog(hog, pos_num,  1)
    hog, neg_reg_descriptors = store.load_region_descriptors_for_hog(hog, neg_num, -1)
        # hog, reg_descriptors = store.load_region_descriptors('hog_Llht0,2_n64_gcFalse_hnt0_bs8x8_cs8x8_ws4,0_da1')
    #     hog, pos_regions, pos_descriptors = load_region_descriptors(pos_descriptor_file)
    #     hog, neg_regions, neg_descriptors = load_region_descriptors(neg_descriptor_file)

    reg_descriptors = list(pos_reg_descriptors) + list(neg_reg_descriptors)
    return reg_descriptors, hog


if __name__ == '__main__':
    # random.seed(123454321) # Use deterministic samples.

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Train a HOG + Linear SVM classifier.')
    parser.add_argument('classifier_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the classifier to train.')
    args = parser.parse_args()

    # Read classifier training file:
    classifier_yaml = training.loadYamlFile(args.classifier_yaml)
    output_dir = args.classifier_yaml.split('.yaml')[0]

    window_dims = tuple(map(int, classifier_yaml['training']['svm']['window_dims']))
    print 'window_dims:', window_dims

    # test_random_with_same_aspect()
    #
    # print 'Test negative generation:'
    # bak_img_dir = classifier_yaml['dataset']['directory']['background']
    # pos_img_dir = classifier_yaml['dataset']['directory']['positive']
    # bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    # exl_info_map = utils.load_opencv_bounding_box_info('/Users/mitchell/data/car-detection/bbinfo/shopping__exclusion.dat')
    # pos_reg_generator = generate_positive_regions(pos_img_dir, bbinfo_dir, classifier_yaml['dataset']['modifiers'], window_dims)
    # # neg_reg_generator = generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims)
    # # all_images = utils.listImagesInDirectory(bak_img_dir)
    # # neg_reg_generator = generate_negative_regions_in_image_with_exclusions(all_images[0], exl_info_map, window_dims)
    # mosaic_gen = utils.mosaic_generator(pos_reg_generator, (4, 6), (window_dims[1], window_dims[0]))
    # # mosaic_gen = utils.mosaic_generator(pos_reg_generator, (20, 30), (40, 60))
    # for mosaic in mosaic_gen:
    #     cv2.imshow('mosaic', mosaic)
    #     while True:
    #         key = cv2.waitKey(1) & 0xFF
    #         if key != 255:
    #             break
    # view_image_regions(neg_reg_generator, window_dims, display_scale=2)

    # Create the hog object with which to compute the descriptors:
    hog = get_hog_object(window_dims)

    db_name = utils.name_from_hog_descriptor(hog)
    svm_name = strutils.safe_name_from_info_dict(classifier_yaml['training']['svm'], 'svm_')
    svm_save_path = 'output/car_detector_{}_{}.yaml'.format(svm_name, db_name)
    if not os.path.isfile(svm_save_path):

        reg_descriptors, hog = create_or_load_descriptors(classifier_yaml, hog, window_dims)

        # Create lists of samples and labels:
        descriptors = [rd.descriptor for rd in reg_descriptors]
        labels = [rd.label for rd in reg_descriptors]

        # Train the classifier:
        train_svm(svm_save_path, descriptors, labels)

    # Test the classifier:
    test_classifier(classifier_yaml, svm_save_path, window_dims)
