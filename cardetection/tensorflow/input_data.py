"""Functions for downloading and reading MNIST data."""
import numpy as np
import tensorflow as tf
import itertools
import random
import cardetection.detection.save_samples as save_samples
import cardetection.carutils.fileutils as fileutils
from progress.bar import Bar as ProgressBar

def batch_shuffle(gen, batch_size=10000):
    while True:
        batch = list(itertools.islice(gen, 0, batch_size))
        random.shuffle(batch)
        for reg in batch:
            yield reg

class DataSet(object):
  def __init__(self, config_yaml_fname, pos_frac):
      config_yaml = fileutils.load_yaml_file(config_yaml_fname)
      self.pos_reg_gen = save_samples.load_positive_region_generator(config_yaml)
      self.neg_reg_gen = batch_shuffle(save_samples.load_negative_region_generator(config_yaml))
      self.window_dims = tuple(map(int, config_yaml['training']['svm']['window_dims']))
      self.pos_frac = pos_frac

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    pos_num = int(batch_size*self.pos_frac)
    neg_num = batch_size - pos_num

    pos_regions = list(itertools.islice(self.pos_reg_gen, 0, pos_num))
    neg_regions = list(itertools.islice(self.neg_reg_gen, 0, neg_num))
    regions = pos_regions + neg_regions

    # Create a tensor containing all images:
    w, h = self.window_dims
    depth = 3
    images = np.zeros((batch_size, h, w, depth), dtype=np.float32)
    progressbar = ProgressBar('Loading regions', max=batch_size)
    for i, reg in enumerate(regions):
        progressbar.next()
        sample = reg.load_cropped_resized_sample(self.window_dims)
        images[i,:,:,:] = sample
    progressbar.finish()

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns*depth]
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])

    # Convert from [0, 255] -> [0.0, 1.0].
    images = np.multiply(images, 1.0 / 255.0)

    # Create one-hot labels:
    pos_labels = np.zeros((pos_num, 2))
    pos_labels[:,0] = 1
    neg_labels = np.zeros((neg_num, 2))
    neg_labels[:,1] = 1
    labels = np.row_stack([pos_labels, neg_labels])

    assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape, labels.shape)
    )

    # Shuffle the data
    perm = np.arange(batch_size)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]

    return images, labels

def read_data_sets(config_yaml_fname, pos_frac=0.5):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(config_yaml_fname, pos_frac)
    # data_sets.validation = DataSet([], 0)
    # data_sets.test = DataSet([], 0)
    return data_sets
