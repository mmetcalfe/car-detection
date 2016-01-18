import os.path
import argparse
import yaml
import numpy as np
import cv2
import cardetection.carutils.images as utils
import cardetection.carutils.geometry as gm
import cardetection.carutils.drawing2d as drawing2d
import cascadetraining as training

def random_colour():
    col = drawing2d.getRandCol()*255
    bs = np.random.uniform(0.5, 1.0)
    return col * bs

def draw_background_object(img, bounding_rect):
    tl = tuple(bounding_rect.tl)
    br = tuple(bounding_rect.br)

    centre = bounding_rect.exact_centre
    rx = centre[0] - bounding_rect.x1
    ry = centre[1] - bounding_rect.y1
    radius = min(rx, ry)
    lw = max(1, int(round(radius*0.1 + np.random.uniform(radius*0.4))))

    col = random_colour()
    cv2.rectangle(img, tl, br, col, lw)

def draw_positive_object(img, bounding_rect):
    centre = tuple(map(int, bounding_rect.exact_centre))

    rx = centre[0] - bounding_rect.x1
    ry = centre[1] - bounding_rect.y1
    # radius = min(rx, ry)
    # lw = max(1, int(round(radius*0.1 + np.random.uniform(radius*0.4))))

    lw = max(1, int(round(ry*0.1 + np.random.uniform(ry*0.25))))
    radius = int(ry*0.5 - lw*0.5)

    col = random_colour()
    cv2.circle(img, centre, radius, col, lw)

def synthesise_background_image(img_dims=(800, 600), num_objects=1000):
    img = np.zeros((img_dims[1],img_dims[0],3), np.uint8)

    for _ in xrange(num_objects):
        # rect = gm.PixelRectangle.random_with_aspect(approx_window_dims, img_dims)
        rect = gm.PixelRectangle.random(img_dims)
        draw_background_object(img, rect)

    return img

def synthesise_positive_image(min_window_dims, objects_per_image=5, img_dims=(800, 600)):
    img = synthesise_background_image(img_dims)

    rects = []
    for _ in xrange(objects_per_image):
        # TODO: Perturb the rectangle.
        rect = gm.PixelRectangle.random_with_aspect(min_window_dims, img_dims)
        draw_positive_object(img, rect)
        rects.append(rect)

    return img, rects

def synthesise_dataset(base_dir, pos_num, neg_num, objects_per_image):
    # Verify that the base directory exists:
    if not os.path.isdir(base_dir):
        raise ValueError('Base directory \'{}\' does not exist.'.format(base_dir))

    # Create the positive directory:
    positive_dir = os.path.join(base_dir, 'positive')
    if not os.path.isdir(positive_dir):
        os.makedirs(positive_dir)

    # Create the background directory:
    background_dir = os.path.join(base_dir, 'background')
    if not os.path.isdir(background_dir):
        os.makedirs(background_dir)

    # # Synthesise background images:
    # for index in xrange(neg_num):
    #     img = synthesise_background_image()
    #
    #     fname = os.path.join(background_dir, '{:06d}.png'.format(index))
    #     cv2.imwrite(fname, img)

    # Synthesise positive images:
    bbinfo_map = {}
    for index in xrange(pos_num):
        img, rects = synthesise_positive_image((150, 100), objects_per_image)

        fname = os.path.join(positive_dir, '{:06d}.png'.format(index))
        bbinfo_map[fname] = rects
        cv2.imwrite(fname, img)

    # Create the bounding box info directory:
    bbinfo_dir = os.path.join(base_dir, 'bbinfo')
    if not os.path.isdir(bbinfo_dir):
        os.makedirs(bbinfo_dir)

    bbinfo_file = os.path.join(bbinfo_dir, 'synthetic__bbinfo.dat')
    utils.save_opencv_bounding_box_info(bbinfo_file, bbinfo_map)


if __name__ == '__main__':
    # random.seed(123454321) # Use deterministic samples.

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset.')
    parser.add_argument('dataset_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file containing the dataset parameters.')
    args = parser.parse_args()

    # Read classifier training file:
    classifier_yaml = training.loadYamlFile(args.dataset_yaml)

    base_dir = classifier_yaml['dataset']['directory']['synthetic']
    pos_num = int(classifier_yaml['training']['svm']['pos_num'])
    neg_num = int(classifier_yaml['training']['svm']['neg_num'])

    synthesise_dataset(base_dir, pos_num, neg_num, objects_per_image=3)
