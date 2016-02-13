import sys
# Ensure that the cardetection package is on the Python path.
sys.path.append('../../')

import flask
from flask import render_template
from flask import request
from flask import jsonify
import cardetection.carutils.fileutils as fileutils

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

if __name__ == '__main__':
    app.run(debug=True)
