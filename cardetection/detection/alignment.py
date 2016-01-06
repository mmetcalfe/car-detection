import cv2
import numpy as np
from progress.bar import Bar as ProgressBar
import os.path

from kitti import getCroppedSampleFromLabel

# From: http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def resizeSample(sample, shape, label):
    # Use INTER_AREA for shrinking and INTER_LINEAR for enlarging:
    targetIsSmaller = shape[1] < (label.x2 - label.x1 + 1) # targetWidth < sampleWidth
    interp = cv2.INTER_AREA if targetIsSmaller else cv2.INTER_LINEAR
    resized = cv2.resize(sample, (shape[1], shape[0]), interpolation=interp)
    return resized

def saveAverageImage(kitti_base, pos_labels, shape, fname, avg_num=None):
    num_images = float(len(pos_labels))
    avg_num = min(avg_num, num_images)
    if avg_num is None:
        avg_num = num_images

    # avg_img = np.zeros((shape[0],shape[1],3), np.float32)
    avg_img = np.zeros(shape, np.float32)
    progressbar = ProgressBar('Averaging ' + fname, max=len(pos_labels))
    num = 0
    for label in pos_labels:
        if num >= avg_num:
            break
        num += 1
        progressbar.next()
        sample = getCroppedSampleFromLabel(kitti_base, label)
        # sample = np.float32(sample)

        resized = resizeSample(sample, shape, label)

        resized = auto_canny(resized)
        resized = np.float32(resized)

        avg_img = cv2.add(avg_img, resized / float(avg_num))
    progressbar.finish()

    cv2.imwrite(fname, avg_img)

def saveSamplesMosaic(kitti_base, pos_labels, mosaicShape, tileShape, fname):
    import random

    imgShape = (mosaicShape[0] * tileShape[0], mosaicShape[1] * tileShape[1])
    mosaic_img = np.zeros((imgShape[0],imgShape[1],3), np.float32)
    # avg_img = np.zeros((imgShape[0],imgShape[1]), np.float32)

    numTiles = mosaicShape[0]*mosaicShape[1]
    labels = pos_labels
    if len(pos_labels) >= numTiles:
        labels = random.sample(pos_labels, numTiles)
    index = 0
    for r in range(mosaicShape[0]):
        for c in range(mosaicShape[1]):
            if index >= len(labels):
                break

            label = labels[index]
            index += 1
            sample = getCroppedSampleFromLabel(kitti_base, label)

            resized = resizeSample(sample, tileShape, label)

            trs = tileShape[0]
            tcs = tileShape[1]
            tr = r * trs
            tc = c * tcs
            mosaic_img[tr:tr+trs, tc:tc+tcs] = resized

    cv2.imwrite(fname, mosaic_img)

# Based on: http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    # print grad.dtype
    # print grad.shape
    return grad

# Based on: http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
def alignImages(imgA, imgB):
    if cv2.__version__ != '3.1.0':
        print 'ERROR: This script requires OpenCV 3.1.0, but cv2.__version__ ==', cv2.__version__
        sys.exit(1)
    # Read 8-bit color image.
    # This is an image in which the three channels are
    # concatenated vertically.

    # im =  cv2.imread("images/emir.jpg", cv2.IMREAD_GRAYSCALE);
    #
    # # Find the width and height of the color image
    # sz = im.shape
    # print sz
    # height = int(sz[0] / 3);
    # width = sz[1]
    #
    # # Extract the three channels from the gray scale image
    # # and merge the three channels into one color image
    # im_color = np.zeros((height,width,3), dtype=np.uint8 )
    # for i in xrange(0,3):
    #     im_color[:,:,i] = im[ i * height:(i+1) * height,:]
    #
    # # Allocate space for aligned image
    # im_aligned = np.zeros((height,width,3), dtype=np.uint8 )

    # The blue and green channels will be aligned to the red channel.
    # So copy the red channel
    # im_aligned[:,:,2] = imgA

    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000,  0.0001)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

    # Get greyscale images:
    greyA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    greyB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Warp the blue and green channels to the red channel
    # Warp imgA to match imgB:
    (cc, warp_matrix) = cv2.findTransformECC(
        get_gradient(greyA),
        get_gradient(greyB),
        warp_matrix, warp_mode
        , criteria)

    width, height, _unused = imgA.shape

    imgAligned = None
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use Perspective warp when the transformation is a Homography
        imgAligned = cv2.warpPerspective(imgB, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use Affine warp when the transformation is not a Homography
        imgAligned = cv2.warpAffine(imgB, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    print 'warp_matrix:', warp_matrix
    return imgAligned

def get_hog(image):
    # winSize = (64,64)
    winSize = (image.shape[1], image.shape[0])
    blockSize = (8,8)
    # blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = [] # (10, 10)# ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist

def partition(values, labels):
    parts = {}

    for i, v in enumerate(values):
        l = labels[i][0]
        if not l in parts:
            parts[l] = []
        parts[l].append(v)

    return parts.values()

def find_label_clusters(kitti_base, kittiLabels, shape, num_clusters, descriptors=None):
    if descriptors is None:
        progressbar = ProgressBar('Computing descriptors', max=len(kittiLabels))
        descriptors = []
        for label in kittiLabels:
            progressbar.next()
            img = getCroppedSampleFromLabel(kitti_base, label)
            # img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            img = resizeSample(img, shape, label)
            hist = get_hog(img)
            descriptors.append(hist)
        progressbar.finish()
    else:
        print 'find_label_clusters,', 'Using supplied descriptors.'
        print len(kittiLabels), len(descriptors)
        assert(len(kittiLabels) == len(descriptors))

    # X = np.random.randint(25,50,(25,2))
    # Y = np.random.randint(60,85,(25,2))
    # Z = np.vstack((X,Y))

    # convert to np.float32
    Z = np.float32(descriptors)

    # define criteria and apply kmeans()
    K = num_clusters
    print 'find_label_clusters,', 'kmeans:', K
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
    # ret,label,center=cv2.kmeans(Z,2,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    print 'ret:', ret
    # print 'label:', label
    # print 'center:', center

    # # Now separate the data, Note the flatten()
    # A = Z[label.ravel()==0]
    # B = Z[label.ravel()==1]

    clusters = partition(kittiLabels, label)
    return clusters
    # # Plot the data
    # from matplotlib import pyplot as plt
    # plt.scatter(A[:,0],A[:,1])
    # plt.scatter(B[:,0],B[:,1],c = 'r')
    # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()
