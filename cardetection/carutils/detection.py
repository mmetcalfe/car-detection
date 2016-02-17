#
# cardetection/carutils/detection.py
#
# Utility functions considered useful only for object detection.
#

import os.path
import glob
import numpy as np
import cv2
import PIL.Image
import cardetection.carutils.geometry as gm
import cardetection.carutils.images as utils

def image_pyramid(img, scale_factor=1.1, min_dims=(32, 32)):
    def shape_is_valid(img):
        h, w = img.shape[:2]
        mw, mh = min_dims
        return h >= mh and w >= mw

    while shape_is_valid(img):
        yield img
        img = utils.resize_sample(img, scale=1.0/scale_factor)


def sliding_window_generator(img, window_dims=(32, 32), scale_factor=1.1, strides=(8, 8), only_rects=False):
    """ Creates an image pyramid, and generates views into each of its levels.
    Returns the pixel rectangle corresponding to each image
    """

    h, w = img.shape[:2]
    orig_dims = (w, h)

    sx, sy = strides
    ww, wh = window_dims
    for img in image_pyramid(img, scale_factor, window_dims):
        h, w = img.shape[:2]
        img_dims = (w, h)
        mx = w - ww
        my = h - wh
        for x in xrange(0, mx, sx):
            for y in xrange(0, my, sy):
                # Create the window in the current pyramid level:
                local_rect = gm.PixelRectangle.from_opencv_bbox([x, y, ww, wh])

                # Crop the window:
                window = None
                if not only_rects:
                    window = utils.crop_rectangle(img, local_rect)

                # Find the window location in the original image:
                orig_rect = local_rect.scale_image(img_dims, orig_dims)

                if (not window is None) and window.shape[0] != wh:
                    print 'ERROR:', (w, h), (x, y), (mx, my), local_rect

                yield (window, orig_rect)
