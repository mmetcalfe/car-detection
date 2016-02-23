# Car Detection Project

## Dependencies:

Note: This project was developed on OS X Yosemite, v10.10.5.
The web app was also tested and deployed on an Ubuntu 15.10 x64 server. Changes will likely be required to make the code work on other operating systems.

For several modules:

  * numpy
  * OpenCV 3.1.0 (build with "extra" contributed modules)
  * pyaml
  * pillow
  * progress
  * matplotlib
  * skimage (scikit-image), for sliding window generation

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

## To build the web-app:

Install node and npm.

Install bower:

    $ npm install -g bower

Install dependencies:

Run in the root directory:

    $ pip install -r requirements.txt

Run in webapp/cardetector:

    $ npm install
    $ bower install

Note: The app structure was based on https://realpython.com/blog/python/the-ultimate-flask-front-end/

### Deploy web-app to a DigitalOcean Ubuntu 15.10 x64 server:

Note: The droplet tested had the following specs.

$20/mo, 2GB Ram, 2 CPUs, 40GB SSD Disk, 3 TB Transfer, New York 2, Ubuntu 15.10 x64

Add this script to User Data when creating the droplet: https://github.com/digitalocean/do_user_scripts/blob/master/Ubuntu-14.04/web-servers/lamp.yml

Note: See the following for a guide on the NodeJS installation below. https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-an-ubuntu-14-04-server

Install the following:

    # sudo apt-get update
    # sudo apt-get install build-essential
    # sudo apt-get install python-dev
    # sudo apt-get install python-pip
    # sudo apt-get install pkg-config
    # sudo apt-get install libopenblas-dev
    # sudo apt-get install gfortran
    # sudo apt-get install libfreetype6-dev
    # sudo apt-get install libpng12-dev
    # sudo apt-get install libjpeg-turbo8-dev
    # sudo apt-get install python-numpy
    # sudo apt-get install python-scipy
    # sudo apt-get install libglfw3-dev
    # sudo apt-get install python-opencv
    # sudo apt-get install git
    <!-- # sudo apt-get install nodejs npm -->
    # sudo wget -qO- \
      https://raw.githubusercontent.com/creationix/nvm/v0.31.0/install.sh \
      | bash # Install nvm
    # source ~/.bashrc
    # nvm install 5.1.0 # Install a recent version of nodejs
    # nvm use 5.1.0
    # nvm alias default 5.1.0 # Ensure that this version is automatically selected when a new session spawns.
    # sudo apt-get install libapache2-mod-wsgi
    # cd /var/www
    # git clone https://github.com/mmetcalfe/car-detection
    # cd /var/www/car-detection
    # pip install -r requirements.txt
    # npm install -g bower
    # npm install -g gulp

    # # Install dependencies and build the app
    # cd /var/www/car-detection/webapp/cardetector
    # npm install
    # bower install --allow-root
    # gulp # (just kill with Ctrl+C once it's done)

Increase the swap space of the droplet to allow tensorflow to build without g++ crashing.

Note: This happens even with `bazel build --jobs=1`.
https://www.digitalocean.com/community/tutorials/how-to-add-swap-on-ubuntu-14-04

    # sudo fallocate -l 8G /swapfile
    # sudo chmod 600 /swapfile
    # sudo mkswap /swapfile
    # sudo swapon /swapfile
    # sudo echo '/swapfile   none    swap    sw    0   0' >> /etc/fstab

Install TensorFlow from source:

Note: See the following. https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html

    # # Install Bazel
    # sudo apt-get install software-properties-common
    # sudo apt-get install openjdk-8-jdk
    # sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
    # wget https://github.com/bazelbuild/bazel/releases/download/0.1.5/bazel-0.1.5-installer-linux-x86_64.sh
    # chmod +x bazel-0.1.5-installer-linux-x86_64.sh
    # ./bazel-0.1.5-installer-linux-x86_64.sh

    # # Install other Dependencies
    # sudo apt-get install python-numpy swig python-dev

    # # Clone the TensorFlow repository
    # cd ~
    # git clone --recurse-submodules https://github.com/tensorflow/tensorflow

    # # Configure the installation (accept defaults)
    # cd ~/tensorflow
    # ./configure

    # # Create the pip package and install
    # # Note: Set jobs to a low number to avoid g++ running out of memory and
    # # crashing (http://stackoverflow.com/a/34399184/3622526).
    # bazel build --jobs=1 -c opt //tensorflow/tools/pip_package:build_pip_package
    # bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    # sudo pip install -U /tmp/tensorflow_pkg/tensorflow-0.7.0-py2-none-any.whl # .whl name may be different

Check that the app starts without error (use Ctrl+C to quit):

    # cd /var/www/car-detection/webapp/cardetector
    # npm test

Copy the virtual host config and enable the virtual host:

Note: See the following for more information.
https://www.digitalocean.com/community/tutorials/how-to-deploy-a-flask-application-on-an-ubuntu-vps

See also: http://flask.pocoo.org/docs/0.10/deploying/mod_wsgi/#creating-a-wsgi-file

    # sudo a2enmod wsgi # Enable wsgi
    # cd /var/www/car-detection/webapp/cardetector
    # sudo cp cardetector.conf /etc/apache2/sites-available/
    # a2ensite cardetector # Enable the virtual host
    # service apache2 restart # Restart Apache to apply changes

Note: Make sure to change the ServerName in `cardetector.conf` to the ip address of your droplet.

### Upload image data to the website:

Assume we have a local directory `images` containing images:

    $ tar cf imgdata.tar images
    $ scp imgdata.tar root@162.243.238.167:~/

Then on the server:

    # mkdir /var/www/car-detection/data/images/
    # mv imgdata.tar /var/www/car-detection/data/images/
    # cd /var/www/car-detection/data/images/
    # tar xf imgdata.tar
    # # Consider changing permissions.
    # # chmod -R 644 images
    # mv images/* . # Move contents of archive into current directory

Create the cache directory for caching detection images:

    # cd /var/www/car-detection/webapp/cardetector
    # mkdir cardetector/static/cache
    # chown www-data cardetector/static/cache

### Upload TensorFlow checkpoint data to the website:

Assume we have a local directory `cnn-train` containing the TensorFlow checkpoint:

    $ tar czf cnn-train.tar cnn-train
    $ scp cnn-train.tar root@162.243.238.167:~/

Then on the server:

    # mkdir /var/www/car-detection/output/
    # mv cnn-train.tar /var/www/car-detection/output/
    # cd /var/www/car-detection/output/
    # tar xzf cnn-train.tar

Create the cache directory for caching detection images:

    # cd /var/www/car-detection/webapp/cardetector
    # mkdir cardetector/static/cache
    # chown www-data cardetector/static/cache


## To train a cascade classifier:

    $ python -m cardetection.detection.perform_experiments

Trains and evaluates a set of classifiers based on the supplied `template.yaml` file.

Note: A directory containing bounding box data files specifying the positive training samples must be present. This can be created using the annotation tool, the ImageNet downloader, or by converting the labels from the KITTI dataset.


## To train a HOG + Linear SVM classifier:

Start the MongoDB driver:

    $ mongod --config /usr/local/etc/mongod.conf

In another terminal, run:

    $ python -m cardetection.detection.trainhog

Trains a single classifier based on the supplied `classifier.yaml` file.

Note: A directory containing bounding box data files specifying the positive training samples must be present. This can be created using the annotation tool, the ImageNet downloader, or by converting the labels from the KITTI dataset.


## To annotate images with bounding boxes:

    $ python -m cardetection.detection.annotation <img_dir> <info_file> [display_width=1500]

## To generate and save samples based on given annotation info files:

    $ python -m cardetection.detection.save_samples

Note: Uses annotation located in the bbinfo directory specified in template.yaml.

Bounding box data will be loaded from all files with names matching `*__bbinfo.dat`, and exclusion data will be loaded from all files matching `*__exclusion.dat`.

## To download images from ImageNet:

    $ python -m download_synset_images_with_info <download_dir>

Downloads all images with bounding boxes from the synsets listed in `parent_words.yaml` to the given directory.


## To save KITTI labels for training:

    $ python -m cardetection.detection.kitti


## To generate a synthetic dataset:

    $ python -m cardetection.detection.syntheticdataset


## Tensorflow:

Note: At present, the stable versions of TensorFlow (0.6.0) don't work with the code for this project, so it was installed from the source on the current master branch (ie. commit 6671d58ff9c60c724582e4e7a4ddaad0a0acda5a).

See the instructions in [the web app deployment section](#deploy-web-app-to-a-digitalocean-ubuntu-1510-x64-server) for how to build TensorFlow from source on Ubuntu.

The current TensorFlow model used is based on the following (it's nearly identical to the CIFAR-10 example):
  * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
  * https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/alexnet.py
  * https://www.tensorflow.org/versions/r0.7/tutorials/deep_cnn/index.html
  * https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html#deep-mnist-for-experts

It was trained with the following setting (at a rate of ~58 examples/second):
  * Trained for 1550 steps
  * pos_frac: 0.2
  * hard_negative_frac: 0.05
  * exclusion_frac=0.05
  * batch_size: 1000
  * learning_rate: 1e-4

To build and run TensorBoard:

    $ cd <tensorflow repository directory> # e.g. /Users/mitchell/code/tensorflow
    $ bazel build tensorflow/tensorboard:tensorboard
    $ bazel-bin/tensorflow/tensorboard/tensorboard --logdir /projects/car-detection/output/cnn-train/

Then visit the displayed url in a web browser.

## Other scripts:

  * `saveclusters.py`: clusters samples using HOG features and K-Means, then saves an average edge-image and a mosaic of 100 samples for each cluster.

  * `aspect_histogram.py`: saves histograms of angles and aspect ratios of samples in the KITTI dataset.

Useful tools:

  * line-profiler for profiling https://github.com/rkern/line_profiler
