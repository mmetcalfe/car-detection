import os.path
import random
import argparse
import yaml
import itertools
import numpy as np
import cv2
import PIL.Image
import cascadetraining as training
import cardetection.carutils.images as utils
import cardetection.carutils.strutils as strutils
import cardetection.carutils.fileutils as fileutils
import cardetection.carutils.geometry as gm
from progress.bar import Bar as ProgressBar
from cardetection.carutils.datastore import DataStore
import cardetection.detection.generate_samples as generate_samples

def save_generated_bbinfo(pos_num, window_dims, pos_dir, bbinfo_dir):
    w, h = window_dims
    bbinfo_file = os.path.join(bbinfo_dir, 'generated__bbinfo.dat')
    with open(bbinfo_file, 'w') as fh:
        info_lines = []
        for index in xrange(pos_num):
            fname = os.path.join(pos_dir, '{:06d}.png'.format(index))
            info_lines.append('{} 1 0 0 {} {}'.format(fname, w, h))
        fh.write('\n'.join(info_lines))

# @profile
def save_regions(reg_gen, num_regions, window_dims, save_dir):
    progressbar = ProgressBar('Saving regions', max=num_regions)
    index = 0
    for img_region in itertools.islice(reg_gen, 0, num_regions):
        fname = os.path.join(save_dir, '{:06d}.png'.format(index))
        index += 1
        sample = img_region.load_cropped_resized_sample(window_dims)
        cv2.imwrite(fname, sample)
        progressbar.next()
    progressbar.finish()

def main():
    # random.seed(123454321) # Use deterministic samples.

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Train a HOG + Linear SVM classifier.')
    parser.add_argument('classifier_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the classifier to train.')
    args = parser.parse_args()

    # Read classifier training file:
    classifier_yaml = fileutils.load_yaml_file(args.classifier_yaml)
    output_dir = args.classifier_yaml.split('.yaml')[0]

    window_dims = tuple(map(int, classifier_yaml['training']['svm']['window_dims']))
    print 'window_dims:', window_dims

    print 'Preview negative generation:'
    print '  [ESC]  Stop viewing negatives'
    print '  [ S ]  Save negative regions to disk'
    neg_num = int(classifier_yaml['training']['svm']['neg_num'])
    neg_output_dir = classifier_yaml['dataset']['directory']['generation']['output']['negative']
    def get_neg_reg_gen():
        # return generate_samples.load_negative_region_generator(classifier_yaml)
        # return generate_samples.load_exclusion_region_generator(classifier_yaml)
        return generate_samples.load_hard_negative_region_generator(classifier_yaml)

    # # TODO: REMOVE THIS CODE:
    # save_regions(get_neg_reg_gen(), neg_num, window_dims, neg_output_dir)
    # sys.exit(1)

    # neg_reg_generator = generate_samples.generate_negative_regions_in_image_with_exclusions(all_images[0], exl_info_map, window_dims)
    # print len(list(neg_reg_generator))
    mosaic_gen = utils.mosaic_generator(get_neg_reg_gen(), (5, 5), (100, 100))
    # mosaic_gen = utils.mosaic_generator(get_neg_reg_gen(), (10, 15), (40, 60))
    stop = False
    for mosaic in mosaic_gen:
        print 'mosaic'
        cv2.imshow('mosaic', mosaic)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop = True
                break
            if key == ord('s'):
                save_regions(get_neg_reg_gen(), neg_num, window_dims, neg_output_dir)
                stop = True
                break
            if key != 255:
                break
        if stop:
            break

    print 'Preview positive generation:'
    print '  [ESC]  Stop viewing positives'
    print '  [ S ]  Save positive regions to disk'
    pos_num = int(classifier_yaml['training']['svm']['pos_num'])
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    pos_output_dir = classifier_yaml['dataset']['directory']['generation']['output']['positive']
    def get_pos_reg_gen():
        return generate_samples.load_positive_region_generator(classifier_yaml)
    mosaic_gen = utils.mosaic_generator(get_pos_reg_gen(), (4, 6), (window_dims[1], window_dims[0]))
    # mosaic_gen = utils.mosaic_generator(pos_reg_generator, (20, 30), (40, 60))
    stop = False
    for mosaic in mosaic_gen:
        print 'mosaic'
        cv2.imshow('mosaic', mosaic)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop = True
                break
            if key == ord('s'):
                save_regions(get_pos_reg_gen(), pos_num, window_dims, pos_output_dir)
                save_generated_bbinfo(pos_num, window_dims, pos_output_dir, bbinfo_dir)
                stop = True
                break
            if key != 255:
                break
        if stop:
            break

if __name__ == '__main__':
    main()
