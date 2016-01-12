#
# aspect_histogram.py
#
#    This script outputs a histogram of the aspect ratios of positive samples
#    from the template_yaml file.
#

import argparse
import subprocess
import glob
import sys
import os
import re
import random
import math
import cv2

import cardetection.detection.cascadetraining as training
from cardetection.carutils.plotting import saveHistogram
from parkinglot import drawing2d
from detection import kitti

if __name__ == "__main__":
    # random.seed(123454321) # Use deterministic samples.

    # # Parse arguments:
    # parser = argparse.ArgumentParser(description='Perform experiments')
    # parser.add_argument('template_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the trials to generate.')
    # args = parser.parse_args()
    #
    # template_yaml = training.loadYamlFile(args.template_yaml)
    #
    # classifier_yaml = template_yaml
    #
    # bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    # global_info = training.loadGlobalInfo(bbinfo_dir)
    #
    # positive_dir = classifier_yaml['dataset']['directory']['positive']
    #
    # all_rects = []
    # for img_path in training.sampleTrainingImages(positive_dir, ['.*'], None, require_bboxes=True, bbinfo_dir=bbinfo_dir):
    #     key = img_path.split('/')[-1]
    #     rects_str = global_info[key]
    #     rects = training.rectanglesFromCacheString(rects_str)
    #     for rect in rects:
    #         all_rects.append(rect)
    #
    #
    # aspects = map(lambda rect: rect.h/float(rect.w), all_rects)

    kitti_base = '/Users/mitchell/data/kitti/'
    category_types = ['Car', 'Van']
    pos_labels = kitti.getPositiveImageLabels(kitti_base, category_types)

    aspects = map(lambda l: l.aspect, pos_labels)
    saveHistogram('aspects.pdf', aspects)
    plt.close()

    # angles = map(lambda l: l.alpha*180.0/np.pi, pos_labels)
    angles = map(lambda l: l.ry*180.0/np.pi, pos_labels)
    # angles = map(lambda l: l.ry, pos_labels)
    saveHistogram('angles.pdf', angles)

    shape = (64, 64)
    aspect = shape[0]/float(shape[1])
    angle = -90
    angleRange = 5
    aspectRange = 0.5

    filtered = pos_labels
    filtered = filter(lambda l: abs(l.ry*180.0/np.pi - angle) < angleRange/2.0, filtered)
    # filtered = filter(lambda l: abs(l.alpha*180.0/np.pi - angle) < angleRange/2.0, filtered)
    filtered = filter(lambda l: abs(l.aspect-aspect) < aspectRange/2.0, filtered)
    fname = 'avg_{}_{}.png'.format(aspect, angle)
    # saveAverageImage(filtered, shape, fname, avg_num=None)
    saveAverageImage(filtered, shape, 'avgPos.png', avg_num=1000)
    print fname
    print 'len(filtered):', len(filtered)
