import sys
# Ensure that the cardetection package is on the Python path.
sys.path.append('../../')

import os.path
import pprint
import flask
from flask import render_template
from flask import request
from flask import jsonify
import cv2
import cardetection.carutils.images as utils
import cardetection.carutils.fileutils as fileutils
from cardetection.detection.detector import ObjectDetector

app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_add_numbers', methods=['POST'])
def add_numbers():
    """Add two numbers server side, ridiculous but well..."""
    a = request.json['a']
    b = request.json['b']
    return jsonify(result=a + b)

@app.route('/_detector_directories', methods=['GET'])
def detector_directories():
    """Load the detector directories from the config file"""
    config_yaml = fileutils.load_yaml_file('detector-config.yaml')
    return jsonify(detector_directories=config_yaml['detector_directories'])
@app.route('/_image_directories', methods=['GET'])
def image_directories():
    """Load the image directories from the config file"""
    config_yaml = fileutils.load_yaml_file('detector-config.yaml')
    return jsonify(image_directories=config_yaml['image_directories'])

def validate_image_directory(imageDir, config_yaml):
    # Ensure that only the allowed image directories are accessed:
    allowed_img_dirs = [entry['value'] for entry in config_yaml['image_directories']]
    if not imageDir in allowed_img_dirs:
        print 'ERROR: The directory \'{}\' is not in the list of image directories.'.format(imageDir)
        # Indicate that this isn't allowed.
        # TODO: Display a better error to the client.
        flask.abort(403) # HTTP status codes: Forbidden
def validate_detector_directory(detectorDir, config_yaml):
    # Ensure that only the allowed detector directories are accessed:
    allowed_detector_dirs = [entry['value'] for entry in config_yaml['detector_directories']]
    if not detectorDir in allowed_detector_dirs:
        print 'ERROR: The directory \'{}\' is not in the list of detector directories.'.format(detectorDir)
        # Indicate that this isn't allowed.
        # TODO: Display a better error to the client.
        flask.abort(403) # HTTP status codes: Forbidden

@app.route('/_update_preview_state', methods=['POST'])
def update_preview_state():
    """Return the new state of the UI given the new settings."""
    print 'update_preview_state'

    print 'request data:'
    pprint.pprint(request.json)

    # Get the inputs:
    currentImgIndex = request.json['currentImgIndex']
    imageDir = request.json['imageDir']
    detectorDir = request.json['detectorDir']
    performDetection = request.json['performDetection']
    returnImage = request.json['returnImage']

    config_yaml = fileutils.load_yaml_file('detector-config.yaml')

    validate_image_directory(imageDir, config_yaml)

    # Get the images:
    image_list = sorted(utils.list_images_in_directory(imageDir))
    num_images = len(image_list)

    if num_images == 0:
        print 'ERROR: The directory \'{}\' contains no images.'.format(imageDir)
        # TODO: Display a better error to the client.
        flask.abort(404) # HTTP status codes: Not Found

    # Find the current image:
    if not currentImgIndex:
        currentImgIndex = 0
    current_img_index = currentImgIndex % num_images
    current_img_path = image_list[current_img_index]
    send_img_path = current_img_path

    # Perform detection:
    detections = []
    if performDetection:
        send_img_path = 'static/tmp-detection-img.jpg'
        save_img_path = 'project/{}'.format(send_img_path)
        detection_img_exists = os.path.isfile(save_img_path)
        if not (returnImage and detection_img_exists):
            validate_detector_directory(detectorDir, config_yaml)

            # detector = ObjectDetector.load_from_directory(detectorDir)
            with ObjectDetector(detectorDir) as detector:
                img = cv2.imread(current_img_path)
                detections, img = detector.detect_objects_in_image(img)
                cv2.imwrite(save_img_path, img)

    previewState = jsonify({
        'numImages' : num_images,
        'currentImgIndex' : current_img_index,
        'currentImgPath' : current_img_path,
        # 'currentImgUrl' : None,
        'detections' : detections
    })

    if returnImage:
        print 'preview state:', previewState.data
        # Return the image
        return flask.send_file(send_img_path)
    else:
        # Return the new preview state:
        print 'response data:', previewState.data
        return previewState

if __name__ == '__main__':
    app.run(debug=True)
