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
import cardetection.carutils.geometry as gm
from progress.bar import Bar as ProgressBar
from cardetection.carutils.datastore import DataStore
import trainhog as trainhog

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

    print 'Test negative generation:'
    bak_img_dir = classifier_yaml['dataset']['directory']['background']
    exl_info_map = utils.load_opencv_bounding_box_info('/Users/mitchell/data/car-detection/bbinfo/shopping__exclusion.dat')
    neg_reg_generator = trainhog.generate_negative_regions_with_exclusions(bak_img_dir, exl_info_map, window_dims, classifier_yaml['dataset']['modifiers'])
    all_images = utils.listImagesInDirectory(bak_img_dir)
    # neg_reg_generator = trainhog.generate_negative_regions_in_image_with_exclusions(all_images[0], exl_info_map, window_dims)
    # print len(list(neg_reg_generator))
    mosaic_gen = utils.mosaic_generator(neg_reg_generator, (10, 15), (40, 60))
    stop = False
    for mosaic in mosaic_gen:
        print 'mosaic'
        cv2.imshow('mosaic', mosaic)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop = True
                break
            if key != 255:
                break
        if stop:
            break

    print 'Test positive generation:'
    bak_img_dir = classifier_yaml['dataset']['directory']['background']
    pos_img_dir = classifier_yaml['dataset']['directory']['positive']
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    exl_info_map = utils.load_opencv_bounding_box_info('/Users/mitchell/data/car-detection/bbinfo/shopping__exclusion.dat')
    pos_reg_generator = trainhog.generate_positive_regions(pos_img_dir, bbinfo_dir, classifier_yaml['dataset']['modifiers'], window_dims)
    mosaic_gen = utils.mosaic_generator(pos_reg_generator, (4, 6), (window_dims[1], window_dims[0]))
    # mosaic_gen = utils.mosaic_generator(pos_reg_generator, (20, 30), (40, 60))
    for mosaic in mosaic_gen:
        cv2.imshow('mosaic', mosaic)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break
    view_image_regions(neg_reg_generator, window_dims, display_scale=2)

    # for img_path in all_images:
    #     if not utils.info_entry_for_image(exl_info_map, img_path):
    #         continue
    #     neg_reg_generator = trainhog.generate_negative_regions_in_image_with_exclusions(img_path, exl_info_map, window_dims)
    #     print len(list(neg_reg_generator))
