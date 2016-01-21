import sys
import os.path
import numpy as np
import cv2
import cardetection.detection.kitti as kitti
import cardetection.detection.alignment as alignment
import cardetection.carutils.geometry as gm

# TODO: Make this a more generic function within the kitti module?
def filterLabels(pos_labels, shape):
    # shape = (64, 64)
    aspect = shape[0]/float(shape[1])
    angle = -90
    angleRange = 10000
    # aspectRange = 0.5
    aspectRange = 1

    filtered = pos_labels
    # # Filter observation angle:
    # filtered = filter(lambda l: abs(l.ry*180.0/np.pi - angle) < angleRange/2.0, filtered)
    # # filtered = filter(lambda l: abs(l.alpha*180.0/np.pi - angle) < angleRange/2.0, filtered)
    #
    # # Filter aspect ratio:
    # filtered = filter(lambda l: abs(l.aspect-aspect) < aspectRange/2.0, filtered)
    #
    # # Filter small samples:
    # minFactor = 0.5
    # minShape = (shape[0] * minFactor, shape[1] * minFactor)
    # filtered = filter(lambda l: l.x2 - l.x1 + 1 >= minShape[0] and l.y2 - l.y1 + 1 >= minShape[1], filtered)

    return filtered

# def displayGradient(imgA):
#     greyA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
#     gradA = get_gradient(greyA)
#     gradA = gradA - np.min(gradA)
#     gradA = gradA / np.max(gradA)
#     cv2.imshow("displayGradient", gradA)
#     cv2.destroyAllWindows()
#     cv2.waitKey(0)

if __name__ == '__main__':
    if cv2.__version__ != '3.1.0':
        print 'ERROR: This script requires OpenCV 3.1.0, but cv2.__version__ ==', cv2.__version__
        sys.exit(1)

    kitti_base = '/Users/mitchell/data/kitti/'

    # shape = (64, 83)
    carShape = (64, 96)
    carAspect = carShape[1]/float(carShape[0])
    shape = carShape

    category_types = ['Car'] #, 'Van']
    pos_labels = kitti.getPositiveImageLabels(kitti_base, category_types)
    filteredLabels = filterLabels(pos_labels, carShape)
    # filteredLabels = filteredLabels[:500]

    print 'Selected {} of {} labels for processing.'.format(len(filteredLabels), len(pos_labels))

    # Modify label bounding boxes:
    for label in filteredLabels:
        rect = gm.Rectangle.fromList(label.opencv_bbox)
        aspectRect = gm.extendBoundingBox(rect, carAspect)
        paddedRect = gm.padBoundingBox(aspectRect, (0.1, 0.1))
        label.opencv_bbox = [paddedRect.x, paddedRect.y, paddedRect.w, paddedRect.h]


    print 'Filtering bad labels...'
    labels = filter(lambda l: kitti.isLabelRectWithinImage(l), filteredLabels)

    print 'Selected {} of {} labels for processing.'.format(len(labels), len(filteredLabels))

    # base_fname = os.path.join(os.path.dirname(__file__), 'output/cluster')

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        print 'ERROR: Output directory "{}" does not exist.'.format(output_dir)
        sys.exit(1)
    else:
        print 'Using output directory "{}".'.format(output_dir)
    base_fname = os.path.join(output_dir, 'cluster')

    num_clusters = 10
    clusters = alignment.find_label_clusters(kitti_base, labels, shape, num_clusters)
    for i, cluster in enumerate(clusters):
        mosaicFname = '{}_{}_mosaic.png'.format(base_fname, i)
        mosaicShape = (10, 10)
        alignment.saveSamplesMosaic(kitti_base, cluster, mosaicShape, shape, mosaicFname)

        avgFname = '{}_{}_avg.png'.format(base_fname, i)
        alignment.saveAverageImage(kitti_base, cluster, shape, avgFname, avg_num=None)

    sys.exit(0)

    # imgA = kitti.getCroppedSampleFromLabel(kitti_base, labels[50])
    # imgB = kitti.getCroppedSampleFromLabel(kitti_base, labels[15])
    #
    # imgA = cv2.resize(imgA, shape, interpolation=cv2.INTER_AREA)
    # imgB = cv2.resize(imgB, shape, interpolation=cv2.INTER_AREA)
    #
    # # # Just make 2 crops of some image
    # # img_path = kitti.imagePath(kitti_base, 3)
    # # img = cv2.imread(img_path)
    # # posA = (50, 50)
    # # posB = (70, 60)
    # # size = (300, 300)
    # # # imgA = img[y1:y2, x1:x2, :]
    # # imgA = img[posA[1]:posA[1]+size[1], posA[0]:posA[0]+size[0], :]
    # # imgB = img[posB[1]:posB[1]+size[1], posB[0]:posB[0]+size[0], :]
    #
    # cv2.imshow("imgA", imgA)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow("imgB", imgB)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # imgAligned = alignImages(imgA, imgB)
    #
    # avgRaw = cv2.addWeighted(imgA, 0.5, imgB, 0.5, 0)
    # avgAligned = cv2.addWeighted(imgA, 0.5, imgAligned, 0.5, 0)
    #
    # # Show final output
    # cv2.imshow("Raw average", avgRaw)
    # cv2.imshow("Aligned average", avgAligned)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
