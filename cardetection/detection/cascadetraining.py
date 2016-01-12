import urllib2
import tarfile
from PIL import Image
import io
import os.path
import yaml
import argparse
import subprocess
import glob
import random
import re
import sys
from pprint import pprint

import cardetection.carutils.geometry as gm
from cardetection.carutils.images import listImagesInDirectory

class TooFewImagesError(Exception):
    def __init__(self, presentCounts, requiredCounts):
        self.presentCounts = presentCounts
        self.requiredCounts = requiredCounts
    def __str__(self):
        return 'TooFewImagesError: ({} < {})'.format(repr(self.presentCounts), repr(self.requiredCounts))

# loadYamlFile :: String -> IO (Tree String)
def loadYamlFile(fname):
    if not os.path.isfile(fname):
        raise ValueError('Input file \'{}\' does not exist!'.format(fname))
    file = open(fname, 'r')
    data = yaml.load(file)
    file.close()
    return data

# loadInfoFile :: String -> Map String String
def loadInfoFile(bbinfo_file):
    bbinfo_cache = {}
    with open(bbinfo_file, 'r') as dat_file:
        for line in dat_file.readlines():
            parts = line.strip().partition(' ')
            image_path = parts[0].split('/')[-1]
            details = parts[2]
            bbinfo_cache[image_path] = details
    return bbinfo_cache

# loadGlobalInfo :: String -> Map String String
def loadGlobalInfo(bbinfo_folder):
    global_info = {}
    cache_files = glob.glob("{}/{}__*.dat".format(bbinfo_folder, '*'))
    for cache_file_name in cache_files:
        info_cache = loadInfoFile(cache_file_name)
        global_info.update(info_cache)
        # with open(cache_file_name, 'r') as dat_file:
        #     for line in dat_file.readlines():
        #         parts = line.strip().partition(' ')
        #         image_path = parts[0].split('/')[-1]
        #         details = parts[2]
        #         global_info[image_path] = details
    return global_info

# requiredImageCounts :: Tree String -> (Int, Int, Int)
def requiredImageCounts(trial_yaml):
    descr_yaml = trial_yaml['dataset']['description']
    if 'number' in descr_yaml and 'posFrac' in descr_yaml and 'hardNegFrac' in descr_yaml:
        datasetSize = int(descr_yaml['number'])
        posFrac = float(descr_yaml['posFrac'])
        hardNegFrac = float(descr_yaml['hardNegFrac'])
        numPos = int(round(datasetSize * posFrac))
        numNonPos = datasetSize - numPos
        numNeg = int(round(numNonPos * hardNegFrac))
        numBak = max(0, numNonPos - numNeg)
        return (numPos, numNeg, numBak)
    elif 'numPositive' in descr_yaml and 'numNegative' in descr_yaml and 'numBackground' in descr_yaml:
        numPositive = descr_yaml['numPositive']
        numNegative = descr_yaml['numNegative']
        numBackground = descr_yaml['numBackground']
        numPos = int(numPositive) if numPositive != 'ALL' else None
        numNeg = int(numNegative) if numNegative != 'ALL' else None
        numBak = int(numBackground) if numBackground != 'ALL' else None
        return (numPos, numNeg, numBak)
    else:
        raise ValueError('Invalid dataset specifications in yaml file.')

# sampleTrainingImages :: String -> [String] -> Int -> [String]
def sampleTrainingImages(image_dir, synsets, sample_size, require_bboxes=False, bbinfo_dir=None):
    image_list = listImagesInDirectory(image_dir)
    # regexString = '.*({})_.*\.jpg'.format('|'.join(synsets))
    # regexString = 'n?({})(_.*)?\.(jpg|png)'.format('|'.join(synsets))
    regexString = '(.*/)?({})(_.*)?\.(jpg|png)'.format('|'.join(synsets))
    synsetProg = re.compile(regexString)

    # Filter image file lists to match sysnet specifications:
    filtered_image_list = filter(lambda x: synsetProg.match(x), image_list)

    # print regexString
    # if len(regexString) > 100:
    #     # print image_list
    #     print filtered_image_list

    # If bounding boxes are required, filter out images without bounding boxes:
    if require_bboxes:
        global_info = loadGlobalInfo(bbinfo_dir)
        bbox_filter = lambda img_path: os.path.split(img_path)[1] in global_info
        filtered_image_list = filter(bbox_filter, filtered_image_list)

    print len(filtered_image_list)

    # Truncate file lists to sample the correct number of images:
    image_sample = filtered_image_list
    if not sample_size is None:
        image_sample = random.sample(filtered_image_list, min(sample_size, len(filtered_image_list)))

    return image_sample

# From: http://stackoverflow.com/a/312464
# chunks :: [Int] -> [[Int]]
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

# rectanglesFromCacheString :: String -> [Rectangle]
def rectanglesFromCacheString(rects_str):
    num, unused_, objs_str = rects_str.partition(' ')

    obj_floats = map(float, objs_str.split(' '))
    # obj_rounded = map(round, obj_floats)
    obj_rounded = map(int, obj_floats) # Note: Rounding can cause invalid bounding boxes.
    obj_ints = map(int, obj_rounded)

    objs = []
    if int(num) > 0:
        objs = list(chunks(obj_ints, 4))

    rects = map(gm.PixelRectangle.from_opencv_bbox, objs)

    assert len(rects) == int(num)
    return rects

# checkBoundingBoxes :: [String] -> String -> IO ()
def checkBoundingBoxes(img_paths, bbinfo_dir):
    global_info = loadGlobalInfo(bbinfo_dir)

    error_occurred = False
    bad_images = []
    warn_images = []
    for img_path in img_paths:
        imsize = None
        try:
            im = Image.open(img_path)
            imsize = im.size
            im.close()
        except Exception as e:
            print "Error opening image {}: {}".format(img_path, e)
            error_occurred = True
            continue

        # Get bounding box:
        key = img_path.split('/')[-1]
        rects_str = global_info[key]
        rects = rectanglesFromCacheString(rects_str)
        for rect in rects:
            lf = rect.x < 0
            tf = rect.y < 0
            rf = rect.x + rect.w > imsize[0]
            bf = rect.y + rect.h > imsize[1]
            sf = rect.w < 1 or rect.h < 1

            extent = (rect.x + rect.w, rect.y + rect.h)

            if rect.x + rect.w == imsize[0] or rect.y + rect.h == imsize[1]:
                warn_images.append((img_path, rect, extent, imsize))

            if lf or tf or rf or bf or sf:
                bad_images.append((img_path, rect, extent, imsize))

    if warn_images:
        print """
WARNING: Some bounding boxes touch the image edges.

    An older version of this script defined bounding boxes differently
    in which case the case where rect.x + rect.w == img_width would
    indicate a bounding box extending outside of the image dimensions.
    The newer KITTI scripts define bounding box size as the true pixel
    size, making this test indicate a box touching the image edge, which
    is fine.

    If the OpenCV scripts fail, this may be the reason.

    (Uncomment the following line to see the suspicious boxes)
        """
        # pprint(warn_images)

    if error_occurred or bad_images:
        print 'EXITING DUE TO INVALID BOUNDING BOXES:'
        pprint(bad_images)
        sys.exit(1)
        assert(False)

# preprocessTrial :: Tree String -> String -> IO ()
def preprocessTrial(classifier_yaml, output_dir):
    # Determine required number of images:
    requiredCounts = requiredImageCounts(classifier_yaml)
    numPos, numNeg, numBak = requiredCounts

    pos_img_dir = classifier_yaml['dataset']['directory']['positive']
    bak_img_dir = classifier_yaml['dataset']['directory']['background']
    neg_img_dir = classifier_yaml['dataset']['directory']['negative']
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    posSets = classifier_yaml['dataset']['description']['synsets']['pos']
    bakSets = classifier_yaml['dataset']['description']['synsets']['bak']
    negSets = classifier_yaml['dataset']['description']['synsets']['neg']

    pos_image_files = sampleTrainingImages(pos_img_dir, posSets, numPos, require_bboxes=True, bbinfo_dir=bbinfo_dir)
    bak_image_files = sampleTrainingImages(bak_img_dir, bakSets, numBak)
    neg_image_files = sampleTrainingImages(neg_img_dir, negSets, numNeg)

    # Complain if there aren't enough images:
    there_are_too_few_images = False
    presentCounts = (len(pos_image_files), len(neg_image_files), len(bak_image_files))
    print presentCounts
    if len(pos_image_files) <= 0 or any(map(lambda (p, r): p < r, zip(presentCounts, requiredCounts))):
        # raise ValueError('Not enough images!')
        raise TooFewImagesError(presentCounts, requiredCounts)

    # Create the output directory:
    if not os.path.isdir(output_dir):
        print '## Creating output directory: {}'.format(output_dir)
        os.makedirs(output_dir)
    else:
        print '## Using existing output directory: {}'.format(output_dir)

    # Print names of images with invalid bounding boxes:
    checkBoundingBoxes(pos_image_files, bbinfo_dir)

    # Load the global info file with bounding boxes for all positive images:
    # global_info_fname = 'info.dat'
    global_info = loadGlobalInfo(bbinfo_dir)

    pos_info_fname = '{}/positive.txt'.format(output_dir)
    neg_info_fname = '{}/negative.txt'.format(output_dir)

    # Note: image paths in the data file have to be relative to the file itself.
    abs_output_dir = os.path.abspath(output_dir)

    def write_img(img, dat_file, write_bbox=True):
        img_path = None
        if os.path.isabs(img):
            img_path = os.path.relpath(img, abs_output_dir)
        else:
            img_path = os.path.relpath(img, output_dir)

        dat_line = img_path
        if write_bbox:
            key = img.split('/')[-1]
            details = global_info[key]
            dat_line = "{} {}".format(img_path, details)

        # print 'dat_line:', dat_line
        dat_file.write(dat_line)

    from random import shuffle
    shuffle(pos_image_files)

    # Write pos_image_files and bounding box info to pos_info_fname:
    with open('{}'.format(pos_info_fname), 'w') as dat_file:
        images = pos_image_files
        write_img(images[0], dat_file)
        for img in images[1:]:
            # Use the bounding boxes from the global info file:
            dat_file.write("\n")
            write_img(img, dat_file)
        dat_file.flush()


    all_neg_image_files = bak_image_files + neg_image_files
    shuffle(all_neg_image_files)
    # Write neg_image_files to neg_info_fname:
    with open('{}'.format(neg_info_fname), 'w') as dat_file:
        images = all_neg_image_files
        write_img(images[0], dat_file, False)
        for img in images[1:]:
            dat_file.write("\n")
            write_img(img, dat_file, False)
        dat_file.flush()

def createSamples(classifier_yaml, output_dir):
    # numPos, _, _ = requiredImageCounts(classifier_yaml)

    balls_vec_fname = '{}/balls.vec'.format(output_dir)
    pos_info_fname = '{}/positive.txt'.format(output_dir)

    numPos = 0
    with open(pos_info_fname, 'r') as fh:
        numPos = sum(1 for line in fh)

    samplesCommand = [ 'opencv_createsamples'
        , '-info', pos_info_fname
        , '-vec',  balls_vec_fname
        , '-num',  str(numPos)
        , '-w',    classifier_yaml['training']['cascade']['sampleSize'][0]
        , '-h',    classifier_yaml['training']['cascade']['sampleSize'][1]
        ]

    print ' '.join(samplesCommand)

    try:
        with open('{}/output_create_samples.txt'.format(output_dir), 'w') as cmd_output_file:
            cmd_ouptut = subprocess.check_call(samplesCommand, stdout=cmd_output_file, stderr=subprocess.STDOUT, cwd='.')
    except subprocess.CalledProcessError as e:
        print 'ERROR:'
        print '\te.returncode: {}'.format(e.returncode)
        print '\te.cmd: {}'.format(e.cmd)
        print '\te.output: {}'.format(e.output)
        sys.exit(0)


# trainClassifier :: Tree String -> String -> IO ()
def trainClassifier(classifier_yaml, output_dir):
    traincascade_data_dir = '{}/data'.format(output_dir)
    if not os.path.isdir(traincascade_data_dir):
        print '## Creating training data directory: {}'.format(traincascade_data_dir)
        os.makedirs(traincascade_data_dir)
    else:
        print '## Using existing training data directory: {}'.format(traincascade_data_dir)

    # # From the opencv_traincascade documentation:
    # #    -numPos <number_of_positive_samples>
    # #    -numNeg <number_of_negative_samples>
    # #        Number of positive/negative samples used in training for every classifier stage.
    # # The key word being 'every'. We need to ensure that we don't ask for so
    # # many samples that the last stages don't have enough.
    # #
    # # We'll use the formula derived on the following page to decide how many
    # # samples to use, after solving for numPos:
    # #     http://answers.opencv.org/question/4368/
    # # (the formula seems a little off to me - it seems like we should raise
    # #  to the power of numStages rather than multiplying? The latter is more
    # #  conservative though, so it shouldn't be a problem.)
    # numPos, numNeg, numBak = requiredImageCounts(classifier_yaml)
    # skipFrac = float(classifier_yaml['training']['boost']['skipFrac']) # A count of all the skipped samples from vec-file (for all stages).
    # skippedSamples = numPos * skipFrac # A count of all the skipped samples from vec-file (for all stages).
    # minHitRate = float(classifier_yaml['training']['boost']['minHitRate'])
    # numStages = float(classifier_yaml['training']['basic']['numStages'])
    # numPosTraining = int((numPos - skippedSamples) / (1 + (1 - minHitRate) * (numStages - 1)))
    # # numPosTraining = int((numPos - skippedSamples) / (1 + (1 - minHitRate)**(numStages - 1))) # ??
    # numNegTraining = (numNeg + numBak)

    balls_vec_fname = '{}/balls.vec'.format(output_dir)
    neg_info_fname = '{}/negative.txt'.format(output_dir)

    trainingCommand = [ 'opencv_traincascade'
        , '-vec',               balls_vec_fname.split('/')[-1]
        , '-data',              'data'
        , '-bg',                neg_info_fname.split('/')[-1]
        , '-numPos',            classifier_yaml['training']['basic']['numPos'] #str(numPosTraining)
        , '-numNeg',            classifier_yaml['training']['basic']['numNeg'] #str(numNegTraining)
        , '-numStages',         classifier_yaml['training']['basic']['numStages']
        , '-featureType',       classifier_yaml['training']['cascade']['featureType']
        , '-w',                 classifier_yaml['training']['cascade']['sampleSize'][0]
        , '-h',                 classifier_yaml['training']['cascade']['sampleSize'][1]
        , '-minHitRate',        classifier_yaml['training']['boost']['minHitRate']
        , '-maxFalseAlarmRate', classifier_yaml['training']['boost']['maxFalseAlarmRate']
        , '-weightTrimRate',    classifier_yaml['training']['boost']['weightTrimRate']
        , '-maxDepth',          classifier_yaml['training']['boost']['maxDepth']
        , '-maxWeakCount',      classifier_yaml['training']['boost']['maxWeakCount']
        , '-mode',              classifier_yaml['training']['haarFeature']['mode']
        ]

    # TODO: Pipe ouptut to file, since this is a long running process.
    with open('{}/output_training.txt'.format(output_dir), 'w') as cmd_output_file:
        cmd_ouptut = subprocess.check_call(trainingCommand, stdout=cmd_output_file, stderr=subprocess.STDOUT, cwd='{}'.format(output_dir))

# runClassifier :: Tree String -> String -> IO ()
import cv2
import numpy as np

def checkImageOrientation(img_path):
    import exifread
    with open(img_path, 'rb') as fh:
        exif_tags = exifread.process_file(fh)
        if not 'Image Orientation' in exif_tags:
            return False

        # 0x0112: ('Orientation', {
        #     1: 'Horizontal (normal)',
        #     2: 'Mirrored horizontal',
        #     3: 'Rotated 180',
        #     4: 'Mirrored vertical',
        #     5: 'Mirrored horizontal then rotated 90 CCW',
        #     6: 'Rotated 90 CW',
        #     7: 'Mirrored horizontal then rotated 90 CW',
        #     8: 'Rotated 90 CCW'
        # }),
        orientation = exif_tags['Image Orientation']
        # print dir(orientation)
        if orientation.field_type == 3:
            return True
        elif orientation.field_type == 1:
            return False
        else:
            print 'UNKNOWN EXIF ORIENTATION VALUE:', orientation.field_type

    return False

def runClassifier(classifier_yaml, output_dir):
    traincascade_data_dir = '{}/data'.format(output_dir)
    classifier_xml = '{}/cascade.xml'.format(traincascade_data_dir)

    if not os.path.isfile(classifier_xml):
        print 'ERROR: Classifier does not exist:', classifier_xml
        sys.exit(1)

    classifier = cv2.CascadeClassifier(classifier_xml)

    for test_dir in classifier_yaml['testing']['directories']:
        test_source_name = test_dir.strip('/').split('/')[-1]

        results_dir = '{}/{}_results'.format(output_dir, test_source_name)
        detections_fname = '{}/{}_detections.dat'.format(output_dir, test_source_name)

        img_list = listImagesInDirectory(test_dir)
        random.shuffle(img_list)

        for img_path in img_list:
            img = cv2.imread(img_path)

            while img.shape[0] > 1024:
                print 'resize:', img_path, img.shape
                img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

            # # Check whether the image is upside-down:
            # if checkImageOrientation(img_path):
            #     print 'Flipped!'
            #     img = cv2.flip(img, -1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            minSize = (img.shape[0] / 50, img.shape[1] / 50)
            cars = classifier.detectMultiScale(
                image=gray,
                # scaleFactor=1.05,
                scaleFactor=1.01,
                minNeighbors=4,
                minSize=minSize,
            )

            print img_path, len(cars)
            if len(cars) > 0:
                for (x,y,w,h) in cars:
                    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    lw = max(2, img.shape[0] / 100)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),lw)
                    # roi_gray = gray[y:y+h, x:x+w]
                    # roi_color = img[y:y+h, x:x+w]

                # img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('img', 500, 500)
                cv2.imshow('img', img)
                # cv2.imshow('img',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def cvDrawRectangle(img, rect, col, lw):
    cv2.rectangle(img,tuple(rect.tl),tuple(rect.br),col, lw)

def viewPositiveSamples(classifier_yaml, output_dir):
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    global_info = loadGlobalInfo(bbinfo_dir)

    positive_dir = classifier_yaml['dataset']['directory']['positive']

    # for img_path in glob.glob("{}/*_*.jpg".format(positive_dir)):
    # for img_path in listImagesInDirectory(positive_dir):

    pos_samples = sampleTrainingImages(positive_dir, ['.*'], None, require_bboxes=True, bbinfo_dir=bbinfo_dir)

    print 'Selected {} positive samples.'.format(len(pos_samples))

    for img_path in pos_samples:
        img = cv2.imread(img_path)

        key = img_path.split('/')[-1]
        rects_str = global_info[key]
        rects = rectanglesFromCacheString(rects_str)

        print img_path, len(rects)
        for rect in rects:
            cvDrawRectangle(img, rect, (255,0,0),2)

            # aspectRect = gm.extendBoundingBox(rect, 83/64.0)
            # cvDrawRectangle(img, aspectRect, (0,255,0),2)
            #
            # paddedRect = gm.padBoundingBox(aspectRect, (0.1, 0.1))
            # cvDrawRectangle(img, paddedRect, (0,0,255),2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# def runClassifier(classifier_yaml, output_dir):
#     traincascade_data_dir = '{}/data'.format(output_dir)
#
#     for input_dir in classifier_yaml['testing']['directories']:
#         input_source_name = input_dir.strip('/').split('/')[-1]
#
#         results_dir = '{}/{}_results'.format(output_dir, input_source_name)
#         # if not os.path.isdir(results_dir):
#         #     print '## Creating results directory: {}'.format(results_dir)
#         #     os.makedirs(results_dir)
#         # else:
#         #     print '## Using existing results directory: {}'.format(results_dir)
#
#         detections_fname = '{}/{}_detections.dat'.format(output_dir, input_source_name)
#
#         runCommand = [ './opencv_runner/build/opencv_runner'
#             , traincascade_data_dir + '/cascade.xml'
#             , detections_fname
#             , input_dir
#             , results_dir
#             , 'false' # Do not save result images
#             ]
#
#         try:
#             with open('{}/output_run_{}.txt'.format(output_dir, input_source_name), 'w') as cmd_output_file:
#                 subprocess.check_call(runCommand, stdout=cmd_output_file, stderr=subprocess.STDOUT, cwd='.')
#         except subprocess.CalledProcessError as e:
#             print 'ERROR:'
#             print '\te.returncode: {}'.format(e.returncode)
#             print '\te.cmd: {}'.format(e.cmd)
#             print '\te.output: {}'.format(e.output)
#             sys.exit(0)


if __name__ == "__main__":
    random.seed(123454321) # Use deterministic samples.

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('classifier_yaml', type=str, nargs='?', default='../classifiers/classifier.yaml', help='Filename of the YAML file describing the classifier to train.')
    args = parser.parse_args()

    # Read classifier training file:
    classifier_yaml = loadYamlFile(args.classifier_yaml)
    output_dir = args.classifier_yaml.split('.yaml')[0]


    # TODO: Consider having preprocessTrial return a map of trial info:
    preprocessTrial(classifier_yaml, output_dir)

    createSamples(classifier_yaml, output_dir)

    print '\n## Training classifier...'

    trainClassifier(classifier_yaml, output_dir)

    print '\n## Running classifier...'

    runClassifier(classifier_yaml, output_dir)

    # print '\n## Calculating statistics...'
    #
    # global_info_fname = 'info.dat'
    #
    # # Note: Need to use the global data file because
    # #       pos_info_fname doesn't have bounding boxes for the test set.
    # statsCommand = [ 'python', 'detection_stats.py'
    #     , detections_fname
    #     , global_info_fname
    #     ]
    # subprocess.call(statsCommand, cwd='.')

    # # subprocess.check_output(['ls'], cwd=base_dir)
