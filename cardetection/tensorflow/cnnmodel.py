import os
import tensorflow as tf
import numpy as np

# Variable creation convenience functions:

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape, name=name)
    return tf.Variable(initial)

# Convolution and pooling convenience functions:
# Note: Boundaries, stride size, and pooling type can be changed.

def conv2d(x, W, name=None, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)
def local_norm(x, lsize=4, name='norm'):
    return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
def max_pool(x, k=3, s=2, name='pool'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                           padding='SAME', name=name)

# From: http://stackoverflow.com/a/33816991/3622526
# See also: https://github.com/tensorflow/tensorflow/issues/908
def conv_mosaic(V, use_col=False):
    """ Reshapes the ouptut of a tf.nn.conv2d layer into a mosaic of images
    representing the output of the convolution. """

    # shape: [col_channels, image_height, image_width, num_filters]
    # i.e. shape: [cc, iy, ix, nf]
    shape = V.get_shape()
    cc = int(shape[0]) if use_col else 1 # col_channels
    iy = int(shape[1]) # image_height
    ix = int(shape[2]) # image_width
    nf = int(shape[3]) # num_filters
    V = tf.slice(V,(0,0,0,0),(cc,-1,-1,-1)) #V[0,...]
    # shape: [cc, iy, ix, nf]

    V = tf.transpose(V,(1, 2, 3, 0))
    # shape: [iy, ix, nf, cc]

    V = tf.reshape(V,(iy,ix,nf*cc))
    # shape: [iy, ix, nf*cc]

    # Padding between images:
    pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
    tx = ix + pad
    ty = iy + pad
    V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
    # shape: [ty, tx, nf*cc]

    # Find appropriate tile-dimensions of the new image:
    cx = int(np.sqrt(nf)) - 1
    while nf%cx != 0:
        cx += 1
    cy = nf // cx
    V = tf.reshape(V,(ty,tx,cy,cx,cc))
    # shape: [ty, tx, cy, cx, cc]

    V = tf.transpose(V,(2,0,3,1,4))
    # shape: [cy, ty, cx, tx, cc]

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*ty,cx*tx,cc))
    # shape: [1, cy*ty, cx*tx, 1]

    return V

def flat_mosaic(V, use_col=False):

    """ Reshapes the ouptut of a readout layer into a mosaic (well, an array) of
    images representing the weights contributing to each class.
    """

    # shape: [num_dense_neurons, NUM_CLASSES]
    # shape: [flat_img_size, num_class_images]
    shape = V.get_shape()
    flat_img_size = int(shape[0]) # flat_img_size
    nc = int(shape[1]) # NUM_CLASSES
    # shape: [flat_img_size, nc]

    # Find appropriate (near-square) dimensions for the images:
    ix = int(np.sqrt(flat_img_size)) - 1
    while flat_img_size%ix != 0:
        ix += 1
    iy = flat_img_size // ix

    # print 'V.get_shape()', V.get_shape()
    V = tf.reshape(V, (iy, ix, nc))
    # shape: [iy, ix, nc]

    # Add padding between images:
    pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
    tx = ix + pad
    ty = iy + pad
    V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
    # shape: [ty, tx, nc]

    # Find appropriate tile-dimensions of the new image:
    cx = nc
    cy = 1
    V = tf.reshape(V,(ty,tx,cy,cx))
    # shape: [ty, tx, cy, cx]

    V = tf.transpose(V,(2,0,3,1))
    # shape: [cy, ty, cx, tx]

    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*ty,cx*tx,1))
    # shape: [1, cy*ty, cx*tx, 1]

    return V

# def conv_mosaic(V, use_col=False):
#     """ Reshapes the ouptut of a tf.nn.conv2d layer into a mosaic of images
#     representing the output of the convolution. """
#
#     # shape: [col_channels, image_height, image_width, num_filters]
#     # i.e. shape: [cc, iy, ix, nf]
#     shape = V.get_shape()
#     cc = int(shape[0]) if use_col else 1
#     iy = int(shape[1])
#     ix = int(shape[2])
#     num_filters = int(shape[3])
#     V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
#     # shape: [1, iy, ix, nf]
#
#     V = tf.reshape(V,(iy,ix,num_filters))
#     # shape: [iy, ix, nf]
#
#     # Padding between images:
#     pad = (int(max(2, min(ix, iy) / 7.0)) // 2) * 2
#     tx += ix + pad
#     ty += iy + pad
#     V = tf.image.resize_image_with_crop_or_pad(V, ty, tx)
#     # shape: [ty, tx, nf]
#
#     cx = 8
#     cy = num_filters // cx
#     V = tf.reshape(V,(ty,tx,cy,cx))
#     # shape: [ty, tx, cy, cx]
#
#     V = tf.transpose(V,(2,0,3,1))
#     # shape: [cy, ty, cx, tx]
#
#     # image_summary needs 4d input
#     V = tf.reshape(V,(1,cy*ty,cx*tx,1))
#     # shape: [1, cy*ty, cx*tx, 1]
#
#     return V

def build_model(x, window_dims, use_colour=True):
    PATCH_SIZE = 5
    NUM_OUT_CHNLS_1 = 64
    NUM_OUT_CHNLS_2 = 64
    NUM_DENSE_NEURONS_1 = 512
    NUM_DENSE_NEURONS_2 = 128
    NUM_CLASSES = 2

    # Placeholder for input images:
    # x = tf.placeholder("float", [None, img_pixels], name='input_images')
    # Build the model:
    # img_w, img_h = 28, 28 # for MNIST
    img_w, img_h = window_dims
    num_col_chnls = 3
    img_pixels = img_w*img_h*num_col_chnls

    x_col_image = tf.reshape(x, [-1, img_h, img_w, num_col_chnls])

    x_image = x_col_image
    if not use_colour:
        num_col_chnls = 1
        x_image = tf.image.rgb_to_grayscale(x_col_image)

    print 'x_col_image.get_shape()', x_col_image.get_shape()
    print 'x_image.get_shape()', x_image.get_shape()

    # First convolutional layer:
    NUM_IN_CHNLS_1 = num_col_chnls

    # Define weights and bias terms for layer 1:
    # NUM_OUT_CHNLS_1 = 32

    W_conv1 = weight_variable([PATCH_SIZE, PATCH_SIZE, NUM_IN_CHNLS_1, NUM_OUT_CHNLS_1], name='W_conv1')
    print 'W_conv1.get_shape()', W_conv1.get_shape()
    conv = conv2d(x_image, W_conv1)
    print 'conv.get_shape()', conv.get_shape()
    b_conv1 = bias_variable([NUM_OUT_CHNLS_1], name='b_conv1')
    # Convolve x_image with weight tensor, add bias, apply ReLU, then max pool:
    h_conv1 = tf.nn.relu(conv + b_conv1, name='h_conv1')
    # h_pool1 = max_pool_2x2(h_conv1, name='h_pool1')

    h_pool1 = max_pool(h_conv1, name='h_pool1')
    print 'h_pool1.get_shape()', h_pool1.get_shape()
    h_norm1 = local_norm(h_pool1, name='h_norm1')
    print 'h_norm1.get_shape()', h_norm1.get_shape()

    # Second convolutional layer:
    W_conv2 = weight_variable([PATCH_SIZE, PATCH_SIZE, NUM_OUT_CHNLS_1, NUM_OUT_CHNLS_2], name='W_conv2')
    b_conv2 = bias_variable([NUM_OUT_CHNLS_2], name='b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
    # h_pool2 = max_pool_2x2(h_conv2, name='h_pool2')
    # print 'h_pool2.get_shape()', h_pool2.get_shape()

    h_norm2 = local_norm(h_conv2, name='h_norm2')
    print 'h_norm2.get_shape()', h_norm2.get_shape()
    h_pool2 = max_pool(h_norm2, name='h_pool2')
    print 'h_pool2.get_shape()', h_pool2.get_shape()

    # Densely connected layer:
    # Note: Image size is now 7x7.
    #       Add a fully connected layer with 1024 neurons to process entire image.
    #       Reshape pooling layer tensor into a batch of vectors, multiply by
    #       weight matrix, add bias, then ReLU.

    h_pool2_shape = h_pool2.get_shape()
    reduced_img_h = int(h_pool2_shape[1])
    reduced_img_w = int(h_pool2_shape[2])
    # num_dense_neurons = 1024

    reduced_img_pixels = reduced_img_h*reduced_img_w
    h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_pixels*NUM_OUT_CHNLS_2], name='h_pool2_flat')

    with tf.variable_scope('dense_1') as scope:
        W_fc1 = weight_variable([reduced_img_pixels*NUM_OUT_CHNLS_2, NUM_DENSE_NEURONS_1], name='W_fc1')
        b_fc1 = bias_variable([NUM_DENSE_NEURONS_1], name='b_fc1')
        print 'W_fc1.get_shape()', W_fc1.get_shape()
        h_fc1 = tf.nn.relu_layer(h_pool2_flat, W_fc1, b_fc1, name=scope.name)

    with tf.variable_scope('dense_2') as scope:
        W_fc2 = weight_variable([NUM_DENSE_NEURONS_1, NUM_DENSE_NEURONS_2], name='W_fc2')
        b_fc2 = bias_variable([NUM_DENSE_NEURONS_2], name='b_fc2')
        print 'W_fc2.get_shape()', W_fc2.get_shape()
        h_fc2 = tf.nn.relu_layer(h_fc1, W_fc2, b_fc2, name=scope.name)

    # Dropout:
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name='h_fc2_drop')

    # Readout layer:
    with tf.variable_scope('softmax_linear') as scope:
        W_ro = weight_variable([NUM_DENSE_NEURONS_2, NUM_CLASSES], name='W_ro')
        b_ro = bias_variable([NUM_CLASSES], name='b_ro')
        print 'W_ro.get_shape()', W_ro.get_shape()

        softmax_linear = tf.nn.xw_plus_b(h_fc2_drop, W_ro, b_ro, name=scope.name)

    # # Add summary ops to collect data
    # w_hist = tf.histogram_summary("W_conv1", W_conv1, name='w_hist')
    # b_hist = tf.histogram_summary("b_conv1", b_conv1, name='b_hist')
    # y_hist = tf.histogram_summary("softmax_linear", softmax_linear, name='y_hist')

    # Image summaries:
    tf.image_summary('input', x_image, max_images=10)
    x_image_flat = tf.slice(x_image,(0,0,0,0),(1,-1,-1,-1))
    x_image_flat = tf.reshape(x_image_flat, [-1, img_w*img_h, num_col_chnls], name='x_image_flat')
    x_image_flat = tf.squeeze(x_image_flat) # remove dimensions of size 1
    tf.image_summary('input channels', flat_mosaic(x_image_flat), max_images=1)

    # W_conv1 is (5, 5, in_chnls, out_chnls) == (h, w, in_chnls, out_chnls)
    # Transpose to (in_chnls, h, w, out_chnls)
    W_conv1_transp = tf.transpose(W_conv1,(2,0,1,3), name='W_conv1_transp')
    tf.image_summary('#1 filters W_conv1_transp', conv_mosaic(W_conv1_transp, use_col=True), max_images=1)
    tf.image_summary('#2 filter_results h_conv1', conv_mosaic(h_conv1), max_images=1)

    W_conv2_transp = tf.transpose(W_conv2,(2,0,1,3), name='W_conv2_transp')
    tf.image_summary('#3 filters W_conv2_transp', conv_mosaic(W_conv2_transp), max_images=1)
    tf.image_summary('#4 filter_results h_conv2', conv_mosaic(h_conv2), max_images=1)

    tf.image_summary('#4 filters W_fc2', flat_mosaic(W_fc2), max_images=1)

    regularised_params = [W_fc1, W_fc2]

    return softmax_linear, keep_prob, regularised_params
