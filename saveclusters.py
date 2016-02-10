import sys
import os.path
import argparse
import itertools
import numpy as np
import cv2
import cardetection.detection.alignment as alignment
import cardetection.detection.trainhog as trainhog
import cardetection.carutils.images as utils
import cardetection.carutils.fileutils as fileutils
import cardetection.carutils.geometry as gm
import cardetection.detection.cascadetraining as training

if __name__ == '__main__':
    if cv2.__version__ != '3.1.0':
        print 'ERROR: This script requires OpenCV 3.1.0, but cv2.__version__ ==', cv2.__version__
        sys.exit(1)

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Cluster positive images and save mosaics and average images for each cluster.')
    parser.add_argument('classifier_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the classifier to train.')
    args = parser.parse_args()

    # Read classifier training file:
    classifier_yaml = fileutils.load_yaml_file(args.classifier_yaml)
    output_dir = args.classifier_yaml.split('.yaml')[0]

    window_dims = tuple(map(int, classifier_yaml['training']['svm']['window_dims']))
    window_shape = (window_dims[1], window_dims[0])
    print 'window_dims:', window_dims
    print 'window_shape:', window_dims

    pos_img_dir = classifier_yaml['dataset']['directory']['positive']
    bbinfo_dir = classifier_yaml['dataset']['directory']['bbinfo']
    def get_pos_generator():
        reg_gen = trainhog.generate_positive_regions(pos_img_dir, bbinfo_dir, modifiers_config=None, window_dims=window_dims)
        # Limit the number of regions to use:
        return itertools.islice(reg_gen, 0, 10000)

    pos_reg_generator = get_pos_generator()
    # mosaic_gen = utils.mosaic_generator(pos_reg_generator, (10, 10), (window_dims[1], window_dims[0]))

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        print 'ERROR: Output directory "{}" does not exist.'.format(output_dir)
        sys.exit(1)
    else:
        print 'Using output directory "{}".'.format(output_dir)
    base_fname = os.path.join(output_dir, 'cluster')

    # Get the HOG object to use for clustering:
    hog = trainhog.get_hog_object(window_dims)

    # Find the clusters:
    num_clusters = 5
    clusters = alignment.find_sample_clusters(pos_reg_generator, window_dims, hog, num_clusters)

    # Save the clusters:
    for i, cluster in enumerate(clusters):
        mosaic_fname = '{}_{}_mosaic.png'.format(base_fname, i)
        mosaic_gen = utils.mosaic_generator(cluster, (10, 10), window_shape)
        cv2.imwrite(mosaic_fname, mosaic_gen.next())

        avg_fname = '{}_{}_avg.png'.format(base_fname, i)
        avg_img = utils.average_image(cluster, window_shape, avg_num=None)
        cv2.imwrite(avg_fname, avg_img)

    # sys.exit(0)
