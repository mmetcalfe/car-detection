# Car Detection Project

## Dependencies:

For several modules:

  * numpy
  * OpenCV 3.1.0 (build with "extra" contributed modules)
  * pyaml
  * pillow
  * matplotlib

For trainhog:

    * MongoDB (https://docs.mongodb.org/manual/administration/install-community/)
    * pymongo

Useful for correcting image orientations:
(see cardetection.detection.cascadetraining.checkImageOrientation())

  * exifread

For ImageNet downloader:

  * beautifulsoup4

For carpark rendering:

  * cyglfw3
  * pyopengl


## To train a cascade classifier:

    $ python -m cardetection.detection.perform_experiments

Trains and evaluates a set of classifiers based on the supplied `template.yaml` file.


## To train a HOG + Linear SVM classifier:

Start the MongoDB driver:

    $ mongod --config /usr/local/etc/mongod.conf

In another terminal, run:

    $ python -m cardetection.detection.trainhog

Trains a single classifier based on the supplied `classifier.yaml` file.


## To annotate images with bounding boxes:

    $ python -m cardetection.detection.annotation


## To save KITTI labels for training:

    $ python -m cardetection.detection.kitti


## To download images from ImageNet:

    $ python -m download_synset_images_with_info <download_dir>

Downloads all images with bounding boxes from the synsets listed in `parent_words.yaml` to the given directory.


## Other scripts:

  * `saveclusters.py`: clusters samples using HOG features and K-Means, then saves an average edge-image and a mosaic of 100 samples for each cluster.

  * `aspect_histogram.py`: saves histograms of angles and aspect ratios of samples in the KITTI dataset.
