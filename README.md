To save KITTI labels for training:

    $ python -m cardetection.detection.kitti

To train a classifier:

    $ python -m cardetection.detection.perform_experiments


Other scripts:

  * `saveclusters.py` clusters samples using HOG features and K-Means, then saves an average edge-image and a mosaic of 100 samples for each cluster.

  * `aspect_histogram.py` saves histograms of angles and aspect ratios of samples in the KITTI dataset.
