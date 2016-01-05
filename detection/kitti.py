#
# kitti.py
#
#	This module contains utilty methods for preparing the KITTI dataset to be
#	used for training an OpenCV cascade classifier.
#
#   Much of this code is directly ported from the matlab scripts in the devkit
#   provided with the KITTI dataset.
#

import os.path
import glob

from enum import Enum
class KittiOcclusion(Enum):
    invalid = -1
    none = 0
    partial = 1
    full = 2
    unknown = 3

class KittiLabel:
    def __init__(self):
        self.type = None
        self.truncation = None
        self.occlusion = None
        self.alpha = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.h = None
        self.w = None
        self.l = None
        self.t = None
        self.ry = None
        self.imageIndex = None

    def __repr__(self):
        return '{KittiLabel ' + \
            '| type: ' + str(self.type) + \
            ', truncation: ' + str(self.truncation) + \
            ', occlusion: ' + str(self.occlusion) + \
            ', alpha: ' + str(self.alpha) + \
            ', x1: ' + str(self.x1) + \
            ', y1: ' + str(self.y1) + \
            ', x2: ' + str(self.x2) + \
            ', y2: ' + str(self.y2) + \
            ', h: ' + str(self.h) + \
            ', w: ' + str(self.w) + \
            ', l: ' + str(self.l) + \
            ', t: ' + str(self.t) + \
            ', ry: ' + str(self.ry) + \
            ', imageIndex: ' + str(self.imageIndex) + '}'

    @property
    def opencv_bbox(self):
        # TODO: Check that other scripts correctly represent width and height of
        # bounding boxes.
        # Note: Must add 1 to the difference, since the ranges are inclusive.
        w = self.x2 - self.x1 + 1
        h = self.y2 - self.y1 + 1
        lst = [self.x1, self.y1, w, h]

        return map(int, lst)

    @property
    def aspect(self):
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return w / float(h)

    # KittiLabel.fromString :: String -> KittiLabel
    @classmethod
    def fromString(cls, label_str):
        tokens = label_str.split(' ')
        label = cls()

        # extract label, truncation, occlusion
        label.type = tokens[0] # 'Car', 'Pedestrian', ...
        label.truncation = float(tokens[1]) # truncated pixel ratio ([0..1])
        label.occlusion = KittiOcclusion(int(tokens[2])) # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
        label.alpha = float(tokens[3]) # object observation angle ([-pi..pi])

        # extract 2D bounding box in 0-based coordinates
        label.x1 = float(tokens[4]) # left
        label.y1 = float(tokens[5]) # top
        label.x2 = float(tokens[6]) # right
        label.y2 = float(tokens[7]) # bottom

        # extract 3D bounding box information
        label.h = float(tokens[8]) # box width
        label.w = float(tokens[9]) # box height
        label.l = float(tokens[10]) # box length
        label.t = map(float, tokens[11:14]) # location (x, y, z)
        label.ry = float(tokens[14]) # yaw angle
        return label

    # KittiLabel.readLabels :: String -> [KittiLabel]
    @classmethod
    def listFromFile(cls, label_path):
        labels = []
        with open(label_path, 'r') as fh:
            for line in fh:
                label = KittiLabel.fromString(line)
                labels.append(label)
        return labels

# labelPath :: String -> Int -> String
def labelPath(kitti_base, img_index):
    kitti_labels = '{}/training/label_2/'.format(kitti_base)
    return os.path.normpath('{}/{:06d}.txt'.format(kitti_labels, img_index))
# imagePath :: String -> Int -> String
def imagePath(kitti_base, img_index):
    kitti_labels = '{}/training/image_2/'.format(kitti_base)
    return os.path.normpath('{}/{:06d}.png'.format(kitti_labels, img_index))

# readLabels :: String -> Int -> [KittiLabel]
def readLabels(kitti_base, img_index):
    label_path = labelPath(kitti_base, img_index)
    labels = KittiLabel.listFromFile(label_path)
    for lab in labels:
        lab.imageIndex = img_index
    return labels

def labelIsPositive(label, category_types):
    correct_type = label.type in category_types

    # truncated pixel ratio ([0..1])
    # correct_truncation = label.truncation < 0.1
    correct_truncation = label.truncation < 0.1

    # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
    # correct_occlusion = label.occlusion.value < 2
    correct_occlusion = label.occlusion.value == 0

    return correct_type and correct_truncation and correct_occlusion

# getCarDetectionPositiveImageIndices :: String -> [String] -> [Int]
def getCarDetectionPositiveImageIndices(kitti_base, category_types):
    kitti_labels = '{}/training/label_2/'.format(kitti_base)
    label_files = glob.glob('{}/*.txt'.format(kitti_labels))

    positive_indices = []
    for label_path in label_files:
        labels = KittiLabel.listFromFile(label_path)

        # ['Cyclist', 'Van', 'Tram', 'Car', 'Misc', 'Pedestrian', 'Truck', 'Person_sitting', 'DontCare']
        is_positive = any(map(lambda lb: labelIsPositive(lb, category_types), labels))

        if is_positive:
            _, fname = os.path.split(label_path)
            num, _ = os.path.splitext(fname)
            num = int(num)
            positive_indices.append(num)

    return positive_indices

# saveOpenCVBoundingBoxInfo :: String -> [String] -> [Int] -> IO ()
def saveOpenCVBoundingBoxInfo(kitti_base, category_types, kitti_indices):
    bbinfo_dir = '{}/bbinfo'.format(kitti_base)
    if not os.path.isdir(bbinfo_dir):
        print 'Creating directory:', bbinfo_dir
        os.makedirs(bbinfo_dir)

    bbinfo_file = '{}/kitti__bbinfo.dat'.format(bbinfo_dir)

    bbinfo_lines = []
    for index in kitti_indices:
        img_path = imagePath(kitti_base, index)
        labels = readLabels(kitti_base, index)
        pos_labels = filter(lambda lb: labelIsPositive(lb, category_types), labels)
        bbox_strings = [' '.join(map(str, l.opencv_bbox)) for l in pos_labels]
        num = len(bbox_strings)
        bbinfo_line = '{} {} {}'.format(img_path, num, ' '.join(bbox_strings))
        bbinfo_lines.append(bbinfo_line)

    with open(bbinfo_file, 'w') as fh:
        fh.write('\n'.join(bbinfo_lines))

def getPositiveImageLabels(kitti_base, category_types):
    pos_indices = getCarDetectionPositiveImageIndices(kitti_base, category_types)

    pos_labels = []

    for index in pos_indices:
        # img_path = imagePath(kitti_base, index)
        index_labels = readLabels(kitti_base, index)
        index_pos_labels = filter(lambda lb: labelIsPositive(lb, category_types), index_labels)
        pos_labels.extend(index_pos_labels)
        # bbox_strings = [' '.join(map(str, l.opencv_bbox)) for l in pos_labels]
        # num = len(bbox_strings)
        # bbinfo_line = '{} {} {}'.format(img_path, num, ' '.join(bbox_strings))
        # bbinfo_lines.append(bbinfo_line)
    return pos_labels

import cv2
def getCroppedSampleFromLabel(kitti_base, label):
    if label.imageIndex is None:
        print 'ERROR: label.imageIndex is None'
        return None
    img_path = imagePath(kitti_base, label.imageIndex)
    img = cv2.imread(img_path)

    x, y, w, h = label.opencv_bbox
    x1 = int(label.x1)
    x2 = int(label.x2)
    y1 = int(label.y1)
    y2 = int(label.y2)
    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cropped = img[y1:y2, x1:x2, :]
    # cv2.imshow("cropped", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped

if __name__ == '__main__':
    kitti_base = '/Users/mitchell/data/kitti/'

    # category_types = ['Car', 'Truck', 'Van']
    category_types = ['Car', 'Van']
    pos_indices = getCarDetectionPositiveImageIndices(kitti_base, category_types)
    saveOpenCVBoundingBoxInfo(kitti_base, category_types, pos_indices)
