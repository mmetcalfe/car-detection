import matplotlib.pyplot as plt

import numpy as np
import skimage
from skimage.feature import hog
import skimage.transform
from skimage import data, color, exposure

# image = skimage.transform.resize(data.astronaut(), (80, 80))
image = skimage.transform.resize(data.astronaut(), (64, 64))

image = color.rgb2gray(image)

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True, normalise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
hog_image_rescaled = hog_image

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.savefig('hogtest.pdf')


image = skimage.transform.resize(data.astronaut(), (512, 512))
image = color.rgb2gray(image)


fdm = sum(fd) / len(fd)
fd = fd - fdm
print len(fd), sum(fd), fd

detector_shape = (64, 64)
window_step = 16
pyramid = skimage.transform.pyramid_gaussian(image, downscale=1.1, sigma=0)
print pyramid
for pyramid_im in pyramid:
    if pyramid_im.shape < detector_shape:
        print 'skip', pyramid_im.shape, '<', detector_shape
        break
    print 'pyramid_im.shape', pyramid_im.shape
    print 'detector_shape', detector_shape
    # print 'pyramid_im.ndim', pyramid_im.ndim
    window_step = max(1, np.floor(pyramid_im.shape[1] / 10.0))
    print 'window_step:', window_step
    windows = skimage.util.view_as_windows(pyramid_im, detector_shape, window_step)
    # print len(windows)
    for window_row in windows:
        print 'row'
        for window in window_row:
            # print 'window.shape', window.shape
            # fd= hog(window, orientations=8, pixels_per_cell=(8, 8),
                                # cells_per_block=(1, 1))
            fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), normalise=True)
            # break
            # ax2.imshow(window, cmap=plt.cm.gray)
            # break
        # break
